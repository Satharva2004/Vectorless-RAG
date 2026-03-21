import { useMemo, useRef, useState, useEffect } from "react";
import { Button } from "./components/ui/button";
import { Textarea } from "./components/ui/textarea";

// Lucide icons
import { Send, Sparkles, Loader2, Bot, User, Brain, ChevronDown } from "lucide-react";

type NodeSelectionResponse = {
  question: string;
  selected_node_ids: string[];
  model: string;
  provider: string;
  reasoning?: string | null;
  assistant_content?: string | null;
  reasoning_details?: unknown;
};

type StreamEvent =
  | { event: "log"; data: any }
  | { event: "candidate"; data: any }
  | { event: "shortlist"; data: any }
  | { event: "reasoning"; data: { text: string; model: string; provider: string } }
  | { event: "reasoning_details"; data: { reasoning_details: unknown; model: string; provider: string } }
  | { event: "selection"; data: any }
  | { event: "citations"; data: { citations: any[]; selected_node_ids: string[] } }
  | { event: "delta"; data: { text: string } }
  | { event: "done"; data: any }
  | { event: "error"; data: { message: string } };

async function* readSse(response: Response): AsyncGenerator<StreamEvent> {
  const reader = response.body?.getReader();
  if (!reader) return;

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    while (true) {
      const sepIndex = buffer.indexOf("\n\n");
      if (sepIndex === -1) break;
      const rawEvent = buffer.slice(0, sepIndex);
      buffer = buffer.slice(sepIndex + 2);

      let eventName = "message";
      let dataLine = "";
      for (const line of rawEvent.split("\n")) {
        if (line.startsWith("event:")) eventName = line.slice(6).trim();
        if (line.startsWith("data:")) dataLine += line.slice(5).trim();
      }

      if (!dataLine) continue;
      try {
        const data = JSON.parse(dataLine);
        yield { event: eventName as StreamEvent["event"], data } as StreamEvent;
      } catch {
        continue;
      }
    }
  }
}

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  reasoning?: string;
  isGenerating?: boolean;
  stage?: string;
};

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [busy, setBusy] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTo({
        top: scrollRef.current.scrollHeight,
        behavior: "smooth",
      });
    }
  }, [messages, busy]);

  const onAsk = async () => {
    const q = inputValue.trim();
    if (!q || busy) return;

    setInputValue("");
    setBusy(true);

    const userMsgId = Date.now().toString();
    const assistantMsgId = (Date.now() + 1).toString();

    setMessages((prev) => [
      ...prev,
      { id: userMsgId, role: "user", content: q },
      { id: assistantMsgId, role: "assistant", content: "", reasoning: "", isGenerating: true, stage: "Routing query..." },
    ]);

    const updateAssistantMessage = (updates: Partial<ChatMessage>) => {
      setMessages((prev) =>
        prev.map((msg) => (msg.id === assistantMsgId ? { ...msg, ...updates } : msg))
      );
    };

    let fullAnswer = "";
    let fullReasoning = "";
    let isThinkingMode = false;

    try {
      const selRes = await fetch(`${import.meta.env.VITE_BACKEND_HOSTED}/api/v1/chat/select-nodes/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: q,
          tree_path: "data/hp1_pageindex_tree.json",
          max_nodes: 3,
        }),
      });

      if (!selRes.ok) throw new Error(await selRes.text());

      let selected: NodeSelectionResponse | null = null;
      for await (const evt of readSse(selRes)) {
        if (evt.event === "reasoning") {
          fullReasoning += evt.data.text;
          updateAssistantMessage({ reasoning: fullReasoning, stage: "Routing..." });
        }
        if (evt.event === "selection") {
          selected = evt.data;
          updateAssistantMessage({ stage: "Generating answer..." });
        }
      }

      if (!selected) throw new Error("Processing failed.");

      const ansRes = await fetch(`${import.meta.env.VITE_BACKEND_HOSTED}/api/v1/chat/generate-answer/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: q,
          node_ids: selected.selected_node_ids,
          tree_path: "data/hp1_pageindex_tree.json",
          assistant_content: selected.assistant_content,
          reasoning_details: selected.reasoning_details,
        }),
      });

      if (!ansRes.ok) throw new Error(await ansRes.text());

      let streamBuffer = "";
      for await (const evt of readSse(ansRes)) {
        if (evt.event === "reasoning") {
          fullReasoning += evt.data.text;
          updateAssistantMessage({ reasoning: fullReasoning, stage: "Thinking..." });
        } else if (evt.event === "delta") {
          const text = evt.data.text || "";
          streamBuffer += text;

          // Simple <think> parser
          while (streamBuffer.includes("<think>")) {
            const startIdx = streamBuffer.indexOf("<think>");
            let endIdx = streamBuffer.indexOf("</think>");

            if (endIdx !== -1) {
              const th = streamBuffer.slice(startIdx + 7, endIdx);
              fullReasoning += th;
              updateAssistantMessage({ reasoning: fullReasoning });
              streamBuffer = streamBuffer.slice(0, startIdx) + streamBuffer.slice(endIdx + 8);
            } else {
              isThinkingMode = true;
              const th = streamBuffer.slice(startIdx + 7);
              fullReasoning += th;
              updateAssistantMessage({ reasoning: fullReasoning, stage: "Thinking..." });
              streamBuffer = streamBuffer.slice(0, startIdx); // Keep buffer before think
              break;
            }
          }

          if (isThinkingMode) {
            let endIdx = text.indexOf("</think>");
            if (endIdx !== -1) {
              isThinkingMode = false;
              // Text after </think> goes to answer
              const rText = text.slice(0, endIdx);
              fullReasoning += rText;

              const ansText = text.slice(endIdx + 8);
              fullAnswer += ansText;
              updateAssistantMessage({ reasoning: fullReasoning, content: fullAnswer, stage: "" });
            } else {
              fullReasoning += text;
              updateAssistantMessage({ reasoning: fullReasoning, stage: "Thinking..." });
            }
          } else if (!streamBuffer.includes("<think>")) {
            fullAnswer += text;
            updateAssistantMessage({ content: fullAnswer, stage: "" });
          }
        }
      }

      updateAssistantMessage({ isGenerating: false, stage: "" });

    } catch (err: any) {
      console.error(err);
      fullAnswer += `\n\n[Error: ${err.message}]`;
      updateAssistantMessage({ content: fullAnswer, isGenerating: false, stage: "" });
    } finally {
      setBusy(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      onAsk();
    }
  };

  return (
    <div className="flex h-screen flex-col bg-background text-foreground font-sans">
      <header className="flex items-center justify-between px-6 py-4 border-b border-white/5 bg-background/50 backdrop-blur-md sticky top-0 z-10">
        <div className="flex items-center gap-2">
          <Sparkles className="h-5 w-5 text-blue-500" />
          <h1 className="text-xl font-bold tracking-tight text-white">Grok Chat</h1>
        </div>
      </header>

      <main
        className="flex-1 overflow-y-auto px-4 py-8 custom-scrollbar"
        ref={scrollRef}
      >
        <div className="mx-auto max-w-3xl space-y-8">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-[50vh] text-center space-y-6 animate-in fade-in zoom-in duration-700">
              <div className="h-20 w-20 rounded-full bg-blue-500/10 flex items-center justify-center border border-blue-500/20 shadow-[0_0_30px_rgba(59,130,246,0.2)]">
                <Bot className="h-10 w-10 text-blue-400" />
              </div>
              <div className="space-y-2">
                <h2 className="text-3xl font-bold bg-gradient-to-r from-white to-white/60 bg-clip-text text-transparent">How can I help you today?</h2>
                <p className="text-muted-foreground max-w-md mx-auto">Ask me anything about Harry Potter and the Sorcerer's Stone.</p>
              </div>
            </div>
          ) : (
            messages.map((msg) => (
              <div
                key={msg.id}
                className={`flex gap-4 animate-in slide-in-from-bottom-4 fade-in duration-500 ${msg.role === "user" ? "flex-row-reverse" : ""}`}
              >
                <div className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full border shadow-sm ${msg.role === "user" ? "bg-blue-600 border-blue-500 text-white" : "bg-panel2/80 border-white/10 text-blue-400"}`}>
                  {msg.role === "user" ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
                </div>

                <div className={`flex flex-col gap-2 max-w-[85%] ${msg.role === "user" ? "items-end" : "items-start"}`}>
                  {msg.role === "assistant" && msg.reasoning && (
                    <details 
                      className="group rounded-2xl rounded-tl-sm bg-white/5 border border-white/10 text-sm text-muted-foreground w-full overflow-hidden [&_summary::-webkit-details-marker]:hidden"
                      open={msg.isGenerating ? true : undefined}
                    >
                      <summary className="flex cursor-pointer items-center justify-between p-3 font-medium text-white/70 hover:bg-white/5 transition-colors select-none">
                        <div className="flex items-center gap-3">
                          {msg.isGenerating ? (
                            <Loader2 className="h-4 w-4 animate-spin text-blue-400" />
                          ) : (
                            <Brain className="h-4 w-4 text-gray-400" />
                          )}
                          <span className="text-[14px]">
                            {msg.isGenerating ? "Thinking..." : "Thought Process"}
                          </span>
                        </div>
                        <ChevronDown className="h-4 w-4 opacity-50 transition-transform duration-300 group-open:-rotate-180" />
                      </summary>
                      <div className="px-4 pb-4 pt-1 whitespace-pre-wrap leading-relaxed opacity-80 border-t border-white/5">
                        <div className="border-l-[3px] border-white/10 pl-4 py-2 mt-2 italic text-[14px] text-gray-300">
                          {msg.reasoning}
                        </div>
                      </div>
                    </details>
                  )}

                  {msg.content && (
                    <div
                      className={`rounded-2xl p-4 text-[15px] leading-relaxed relative group ${msg.role === "user"
                        ? "bg-blue-600 text-white rounded-tr-sm shadow-[0_0_20px_rgba(37,99,235,0.15)]"
                        : "bg-panel2/40 border border-white/5 text-white/90 rounded-tl-sm shadow-sm"
                        }`}
                    >
                      <div className="whitespace-pre-wrap">{msg.content}</div>
                    </div>
                  )}

                  {msg.isGenerating && msg.stage && !msg.content && !msg.reasoning && (
                    <div className="flex items-center gap-2 text-sm text-muted-foreground py-2 px-1">
                      <Loader2 className="h-4 w-4 animate-spin text-blue-400" />
                      <span className="animate-pulse">{msg.stage}</span>
                    </div>
                  )}
                </div>
              </div>
            ))
          )}
        </div>
      </main>

      <div className="mx-auto w-full max-w-3xl p-4 sticky bottom-0 bg-background/80 backdrop-blur-xl">
        <div className="relative flex items-end overflow-hidden rounded-2xl border border-white/10 bg-panel2/50 shadow-sm focus-within:ring-1 focus-within:ring-blue-500/50 focus-within:border-blue-500/50 transition-all duration-300">
          <Textarea
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Send a message..."
            className="min-h-[60px] max-h-[200px] w-full resize-none bg-transparent border-0 focus-visible:ring-0 p-4 text-white text-[15px] placeholder:text-muted-foreground custom-scrollbar"
            rows={1}
            disabled={busy}
          />
          <div className="absolute bottom-3 right-3 flex items-center justify-center">
            <Button
              onClick={onAsk}
              disabled={!inputValue.trim() || busy}
              className={`h-9 w-9 rounded-full transition-all duration-300 shadow-md p-0 ${inputValue.trim() && !busy ? "bg-blue-600 hover:bg-blue-500 text-white" : "bg-white/5 text-white/30 hover:bg-white/5"}`}
            >
              {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
            </Button>
          </div>
        </div>
        <div className="text-center mt-3 text-xs text-muted-foreground opacity-70">
          Vectorless RAG · Uses OpenRouter with Grok
        </div>
      </div>

      <style>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 6px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: transparent;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background-color: rgba(255, 255, 255, 0.1);
          border-radius: 20px;
        }
      `}</style>
    </div>
  );
}
