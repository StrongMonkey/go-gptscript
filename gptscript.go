package gptscript

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

var (
	serverProcessMap map[string]*serverEntry

	lock sync.Mutex
)

type serverEntry struct {
	serverProcess       *exec.Cmd
	serverProcessCancel context.CancelFunc
	serverURL           string
	gptscriptCount      int
}

func init() {
	serverProcessMap = make(map[string]*serverEntry)
}

const relativeToBinaryPath = "<me>"

type GPTScript interface {
	Run(context.Context, string, Options) (*Run, error)
	Evaluate(context.Context, Options, ...ToolDef) (*Run, error)
	Parse(context.Context, string) ([]Node, error)
	ParseTool(context.Context, string) ([]Node, error)
	Version(context.Context) (string, error)
	Fmt(context.Context, []Node) (string, error)
	ListTools(context.Context) (string, error)
	ListModels(context.Context) ([]string, error)
	Confirm(context.Context, AuthResponse) error
	PromptResponse(context.Context, PromptResponse) error
	Close()
}

type gptscript struct {
	url    string
	entry  *serverEntry
	hashID string
}

func NewGPTScript(opts GlobalOptions) (GPTScript, error) {
	lock.Lock()
	defer lock.Unlock()
	if _, ok := serverProcessMap[opts.HashID]; !ok {
		serverProcessMap[opts.HashID] = &serverEntry{}
	}
	serverProcessMap[opts.HashID].gptscriptCount++
	entry := serverProcessMap[opts.HashID]

	disableServer := os.Getenv("GPTSCRIPT_DISABLE_SERVER") == "true"

	if entry.serverURL == "" && disableServer {
		entry.serverURL = os.Getenv("GPTSCRIPT_URL")
	}

	if entry.serverProcessCancel == nil && !disableServer {
		if entry.serverURL == "" {
			l, err := net.Listen("tcp", "127.0.0.1:0")
			if err != nil {
				slog.Debug("failed to start gptscript listener", "err", err)
				return nil, fmt.Errorf("failed to start gptscript: %w", err)
			}

			entry.serverURL = l.Addr().String()
			if err = l.Close(); err != nil {
				slog.Debug("failed to close gptscript listener", "err", err)
				return nil, fmt.Errorf("failed to start gptscript: %w", err)
			}
		}

		ctx, cancel := context.WithCancel(context.Background())

		in, _ := io.Pipe()
		entry.serverProcess = exec.CommandContext(ctx, getCommand(), "sys.sdkserver", "--listen-address", entry.serverURL)
		if opts.Env == nil {
			opts.Env = os.Environ()
		}
		entry.serverProcess.Env = append(opts.Env[:], opts.toEnv()...)
		entry.serverProcess.Stdin = in
		entry.serverProcess.Stdout = os.Stdout

		entry.serverProcessCancel = func() {
			cancel()
			_ = in.Close()
		}

		if err := entry.serverProcess.Start(); err != nil {
			entry.serverProcessCancel()
			return nil, fmt.Errorf("failed to start server: %w", err)
		}

		timeoutCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
		defer cancel()
		if err := waitForServerReady(timeoutCtx, entry.serverURL); err != nil {
			entry.serverProcessCancel()
			_ = entry.serverProcess.Wait()
			return nil, fmt.Errorf("failed to wait for gptscript to be ready: %w", err)
		}
	}
	return &gptscript{url: "http://" + entry.serverURL, entry: entry, hashID: opts.HashID}, nil
}

func waitForServerReady(ctx context.Context, serverURL string) error {
	for {
		resp, err := http.Get("http://" + serverURL + "/healthz")
		if err != nil {
			slog.DebugContext(ctx, "waiting for server to become ready")
		} else {
			_ = resp.Body.Close()

			if resp.StatusCode == http.StatusOK {
				return nil
			}
		}

		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(time.Second):
		}
	}
}

func (g *gptscript) Close() {
	lock.Lock()
	defer lock.Unlock()
	g.entry.gptscriptCount--

	if g.entry.gptscriptCount == 0 && g.entry.serverProcessCancel != nil {
		g.entry.serverProcessCancel()
		g.entry.serverProcessCancel = nil
		_ = g.entry.serverProcess.Wait()
		delete(serverProcessMap, g.hashID)
	}
}

func (g *gptscript) Evaluate(ctx context.Context, opts Options, tools ...ToolDef) (*Run, error) {
	return (&Run{
		url:         g.url,
		requestPath: "evaluate",
		state:       Creating,
		opts:        opts,
		tools:       tools,
	}).NextChat(ctx, opts.Input)
}

func (g *gptscript) Run(ctx context.Context, toolPath string, opts Options) (*Run, error) {
	return (&Run{
		url:         g.url,
		requestPath: "run",
		state:       Creating,
		opts:        opts,
		toolPath:    toolPath,
	}).NextChat(ctx, opts.Input)
}

// Parse will parse the given file into an array of Nodes.
func (g *gptscript) Parse(ctx context.Context, fileName string) ([]Node, error) {
	out, err := g.runBasicCommand(ctx, "parse", map[string]any{"file": fileName})
	if err != nil {
		return nil, err
	}

	var doc Document
	if err = json.Unmarshal([]byte(out), &doc); err != nil {
		return nil, err
	}

	for _, node := range doc.Nodes {
		node.TextNode.process()
	}

	return doc.Nodes, nil
}

// ParseTool will parse the given string into a tool.
func (g *gptscript) ParseTool(ctx context.Context, toolDef string) ([]Node, error) {
	out, err := g.runBasicCommand(ctx, "parse", map[string]any{"content": toolDef})
	if err != nil {
		return nil, err
	}

	var doc Document
	if err = json.Unmarshal([]byte(out), &doc); err != nil {
		return nil, err
	}

	for _, node := range doc.Nodes {
		node.TextNode.process()
	}

	return doc.Nodes, nil
}

// Fmt will format the given nodes into a string.
func (g *gptscript) Fmt(ctx context.Context, nodes []Node) (string, error) {
	for _, node := range nodes {
		node.TextNode.combine()
	}

	out, err := g.runBasicCommand(ctx, "fmt", Document{Nodes: nodes})
	if err != nil {
		return "", err
	}

	return out, nil
}

// Version will return the output of `gptscript --version`
func (g *gptscript) Version(ctx context.Context) (string, error) {
	out, err := g.runBasicCommand(ctx, "version", nil)
	if err != nil {
		return "", err
	}

	return out, nil
}

// ListTools will list all the available tools.
func (g *gptscript) ListTools(ctx context.Context) (string, error) {
	out, err := g.runBasicCommand(ctx, "list-tools", nil)
	if err != nil {
		return "", err
	}

	return out, nil
}

// ListModels will list all the available models.
func (g *gptscript) ListModels(ctx context.Context) ([]string, error) {
	out, err := g.runBasicCommand(ctx, "list-models", nil)
	if err != nil {
		return nil, err
	}

	return strings.Split(strings.TrimSpace(out), "\n"), nil
}

func (g *gptscript) Confirm(ctx context.Context, resp AuthResponse) error {
	_, err := g.runBasicCommand(ctx, "confirm/"+resp.ID, resp)
	return err
}

func (g *gptscript) PromptResponse(ctx context.Context, resp PromptResponse) error {
	_, err := g.runBasicCommand(ctx, "prompt-response/"+resp.ID, resp.Responses)
	return err
}

func (g *gptscript) runBasicCommand(ctx context.Context, requestPath string, body any) (string, error) {
	run := &Run{
		url:          g.url,
		requestPath:  requestPath,
		state:        Creating,
		basicCommand: true,
	}

	if err := run.request(ctx, body); err != nil {
		return "", err
	}

	out, err := run.Text()
	if err != nil {
		return "", err
	}
	if run.err != nil {
		return run.ErrorOutput(), run.err
	}

	return out, nil
}

func getCommand() string {
	if gptScriptBin := os.Getenv("GPTSCRIPT_BIN"); gptScriptBin != "" {
		if len(os.Args) == 0 {
			return gptScriptBin
		}
		return determineProperCommand(filepath.Dir(os.Args[0]), gptScriptBin)
	}

	return "gptscript"
}

// determineProperCommand is for testing purposes. Users should use getCommand instead.
func determineProperCommand(dir, bin string) string {
	if !strings.HasPrefix(bin, relativeToBinaryPath) {
		return bin
	}

	bin = filepath.Join(dir, strings.TrimPrefix(bin, relativeToBinaryPath))
	if !filepath.IsAbs(bin) {
		bin = "." + string(os.PathSeparator) + bin
	}

	slog.Debug("Using gptscript binary: " + bin)
	return bin
}
