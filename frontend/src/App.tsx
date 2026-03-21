import { useRef, useState } from "react";
import {
  Send,
  Sparkles,
  BookOpen,
  Loader2,
  ChevronDown,
  BrainCircuit,
  Database,
} from "lucide-react";

// ─── Types ────────────────────────────────────────────────────────────────────

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
  | { event: "reasoning"; data: { text: string; model?: string; provider?: string } }
  | { event: "selection"; data: any }
  | { event: "citations"; data: any }
  | { event: "delta"; data: { text: string } }
  | { event: "done"; data: any }
  | { event: "error"; data: { message: string } };

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  reasoning?: string;
  thinkSeconds?: number;
  isGenerating?: boolean;
  stage?: string;
};

// ─── SSE Reader ───────────────────────────────────────────────────────────────

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

// ─── App ──────────────────────────────────────────────────────────────────────

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [busy, setBusy] = useState(false);
  const [expandedReasoning, setExpandedReasoning] = useState<Record<string, boolean>>({});
  const scrollRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    setTimeout(() => {
      if (scrollRef.current) {
        scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
      }
    }, 50);
  };

  const updateAssistantMessage = (id: string, updates: Partial<ChatMessage>) => {
    setMessages((prev) =>
      prev.map((msg) => (msg.id === id ? { ...msg, ...updates } : msg))
    );
    scrollToBottom();
  };

  const executeSearch = (query: string) => {
    setInputValue(query);
    // Use setTimeout to ensure state is updated before running onAsk logic
    setTimeout(() => {
      submitQuery(query);
    }, 10);
  };

  const submitQuery = async (q: string) => {
    if (!q || busy) return;

    setInputValue("");
    setBusy(true);

    const userMsgId = `u-${Date.now()}`;
    const assistantMsgId = `a-${Date.now()}`;
    const startTime = Date.now();

    setMessages((prev) => [
      ...prev,
      { id: userMsgId, role: "user", content: q },
      { id: assistantMsgId, role: "assistant", content: "", reasoning: "", isGenerating: true, stage: "Routing query..." },
    ]);
    scrollToBottom();

    let fullAnswer = "";
    let fullReasoning = "";

    try {
      // ── Step 1: Node Selection ────────────────────────────────────────────
      const selRes = await fetch(
        `${import.meta.env.VITE_BACKEND_HOSTED}/api/v1/chat/select-nodes/stream`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: q, max_nodes: 3 }),
        }
      );
      if (!selRes.ok) throw new Error(await selRes.text());

      let selected: NodeSelectionResponse | null = null;
      for await (const evt of readSse(selRes)) {
        if (evt.event === "reasoning") {
          fullReasoning += evt.data.text;
          updateAssistantMessage(assistantMsgId, { reasoning: fullReasoning, stage: "Thinking..." });
        }
        if (evt.event === "selection") {
          selected = evt.data;
          updateAssistantMessage(assistantMsgId, { stage: "Reading chapters..." });
        }
      }
      if (!selected) throw new Error("Node selection failed.");

      // ── Step 2: Answer Generation ─────────────────────────────────────────
      const ansRes = await fetch(
        `${import.meta.env.VITE_BACKEND_HOSTED}/api/v1/chat/generate-answer/stream`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            question: q,
            node_ids: selected.selected_node_ids,
            assistant_content: selected.assistant_content,
            reasoning_details: selected.reasoning_details,
          }),
        }
      );
      if (!ansRes.ok) throw new Error(await ansRes.text());

      // Add divider between routing and answer reasoning
      if (fullReasoning) fullReasoning += "\n\n---\n\n";

      for await (const evt of readSse(ansRes)) {
        if (evt.event === "reasoning") {
          fullReasoning += evt.data.text;
          updateAssistantMessage(assistantMsgId, { reasoning: fullReasoning, stage: "Thinking..." });
        } else if (evt.event === "delta") {
          const text = evt.data.text || "";
          const stripped = text.replace(/<\/?think>/g, "");
          if (stripped) {
            fullAnswer += stripped;
            updateAssistantMessage(assistantMsgId, { content: fullAnswer, stage: "" });
          }
        }
      }

      const thinkSeconds = Math.round((Date.now() - startTime) / 1000);
      updateAssistantMessage(assistantMsgId, {
        isGenerating: false,
        stage: "",
        thinkSeconds,
      });
    } catch (err: any) {
      console.error(err);
      updateAssistantMessage(assistantMsgId, {
        content: fullAnswer + `\n\n[Error: ${err.message}]`,
        isGenerating: false,
        stage: "",
      });
    } finally {
      setBusy(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submitQuery(inputValue.trim());
    }
  };

  const toggleReasoning = (id: string) => {
    setExpandedReasoning((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  return (
    <div className="flex items-center justify-center h-screen p-12 md:p-8 overflow-hidden bg-transparent">
      {/* ── Main App Container ── */}
      <div className="flex flex-col w-full max-w-[1800px] h-full rounded-[2rem] overflow-hidden bg-white/95 backdrop-blur-3xl shadow-app-container border border-white/60 relative">

        {/* Header */}
        <header className="flex justify-center items-center px-8 h-20 shrink-0 z-10 border-b border-gray-100/50">
          <div className="flex items-center gap-2 text-[14px] font-bold text-gray-900 tracking-tight">
            <BookOpen className="w-4 h-4 text-orange-500" />
            Harry Potter Vectorless RAG <span className="text-gray-300 font-normal">|</span> <span className="text-gray-500 font-medium">DeepSeek R1</span>
          </div>
        </header>

        {/* Messages Background Gradient */}
        <div className="absolute inset-0 z-0 overflow-hidden pointer-events-none opacity-50">
          <div className="absolute -top-[10%] -right-[10%] w-[1200px] h-[1200px] bg-amber-100/30 rounded-full blur-[160px]" />
          <div className="absolute top-[20%] -left-[10%] w-[1000px] h-[1000px] bg-pink-100/30 rounded-full blur-[140px]" />
          <div className="absolute -bottom-[20%] right-[10%] w-[1100px] h-[1100px] bg-sky-100/40 rounded-full blur-[150px]" />
        </div>

        {/* Messages Area */}
        <div ref={scrollRef} className="flex-1 overflow-y-auto custom-scrollbar px-6 lg:px-24 py-6 z-10 relative">
          <div className="max-w-3xl mx-auto flex flex-col min-h-full">

            {messages.length === 0 ? (
              /* Empty state */
              <div className="flex flex-col items-center justify-center flex-1 animate-in fade-in duration-700 mt-0">
                <h1 className="text-[34px] md:text-[40px] font-bold tracking-tight text-center text-gray-900 mb-14 leading-[1.15]">
                  Ready to explore<br />the Wizarding World?
                </h1>


              </div>
            ) : (
              /* Chat view */
              <div className="space-y-8 pb-4">
                {messages.map((msg) => (
                  <div key={msg.id} className="msg-enter flex flex-col">
                    {msg.role === "user" ? (
                      <div className="flex justify-end">
                        <div className="max-w-[85%] bg-white border border-gray-100 px-5 py-3.5 rounded-[22px] rounded-br-sm shadow-[0_2px_10px_rgba(0,0,0,0.02)] text-[15px] text-gray-800 font-medium">
                          {msg.content}
                        </div>
                      </div>
                    ) : (
                      <div className="flex gap-4">
                        <div className="shrink-0 w-[36px] h-[36px] rounded-full bg-[#f4f7fa] flex items-center justify-center border border-gray-100 shadow-[0_2px_10px_rgba(0,0,0,0.02)] mt-1">
                          <Sparkles className="w-[16px] h-[16px] text-gray-600" />
                        </div>

                        <div className="flex-1 min-w-0 space-y-2.5 pt-1.5">

                          {/* Thinking Block */}
                          {msg.reasoning && (
                            <div className="mb-3">
                              <button
                                onClick={() => !msg.isGenerating && toggleReasoning(msg.id)}
                                className="flex items-center gap-2 text-[12px] font-semibold text-gray-400 hover:text-gray-600 transition-colors"
                              >
                                <Sparkles className="w-[12px] h-[12px]" />
                                {msg.isGenerating ? "Thinking..." : `Reasoned for ${msg.thinkSeconds ?? 5} sec`}
                                {!msg.isGenerating && (
                                  <ChevronDown className={`w-3.5 h-3.5 transition-transform ${expandedReasoning[msg.id] ? "-rotate-180" : ""}`} />
                                )}
                              </button>
                              {(msg.isGenerating || expandedReasoning[msg.id]) && (
                                <div className="mt-3 text-[13px] text-gray-500 bg-white border border-[#eef2f5] rounded-2xl p-5 shadow-sm max-h-[300px] overflow-y-auto custom-scrollbar leading-[1.6]">
                                  {msg.reasoning}
                                </div>
                              )}
                            </div>
                          )}

                          {/* Stage indicator */}
                          {msg.isGenerating && msg.stage && !msg.content && !msg.reasoning && (
                            <div className="flex items-center gap-2 text-[13px] font-medium text-gray-500">
                              <span className="relative flex h-2 w-2">
                                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-gray-400 opacity-75"></span>
                                <span className="relative inline-flex rounded-full h-2 w-2 bg-gray-500"></span>
                              </span>
                              {msg.stage}
                            </div>
                          )}

                          {/* Message Content */}
                          {msg.content && (
                            <div className={`text-[15px] text-gray-900 font-medium leading-[1.7] ${msg.isGenerating ? "cursor-blink" : ""}`}>
                              {msg.content.split('\n').map((line, i) => (
                                <span key={i}>
                                  {line}
                                  {i !== msg.content.split('\n').length - 1 && <br />}
                                </span>
                              ))}
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* ── Input Area ── */}
        <div className="px-6 lg:px-24 pb-8 pt-2 shrink-0 z-20 relative flex flex-col items-center">

          {/* Input Container */}
          <div className="w-full max-w-[800px] bg-white rounded-[24px] shadow-[0_10px_40px_-10px_rgba(0,0,0,0.06)] border border-[#eef2f5] p-3 focus-within:ring-4 focus-within:ring-[#eef2f5] transition-all flex flex-col">

            <div className="flex items-end gap-2 px-1">
              <textarea
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask about Harry Potter and the Philosopher's Stone..."
                disabled={busy}
                className="flex-1 resize-none bg-transparent border-0 outline-none text-[15px] font-medium text-gray-800 placeholder:text-gray-400 placeholder:font-normal py-3 px-3 min-h-[46px] max-h-[200px] custom-scrollbar"
                rows={1}
                onInput={(e) => {
                  const t = e.currentTarget;
                  t.style.height = "auto";
                  t.style.height = Math.min(t.scrollHeight, 200) + "px";
                }}
              />

              <div className="flex items-center gap-2 h-[44px]">
                <button
                  onClick={() => submitQuery(inputValue.trim())}
                  disabled={!inputValue.trim() || busy}
                  className={`w-[44px] h-[44px] rounded-full flex items-center justify-center transition-all shrink-0 ${inputValue.trim() && !busy
                    ? "bg-[#1e1f24] text-white hover:bg-black shadow-md"
                    : "bg-[#eef2f5] text-gray-400"
                    }`}
                >
                  {busy ? <Loader2 className="w-[18px] h-[18px] animate-spin" /> : <Send className="w-[18px] h-[18px] -ml-[2px]" strokeWidth={2.5} />}
                </button>
              </div>
            </div>

            {/* Bottom Context-Aware Action Pills */}
            <div className="flex items-center gap-2.5 mt-2 px-2 pb-1 overflow-x-auto custom-scrollbar">
              {[
                { icon: BookOpen, text: "The Boy Who Lived", prompt: "Summarize the first chapter." },
                { icon: BrainCircuit, text: "Diagon Alley", prompt: "Explain Harry's trip to Diagon Alley." },
                { icon: Database, text: "Characters", prompt: "Who are Ron and Hermione?" },
              ].map((btn, i) => (
                <button
                  key={i}
                  onClick={() => executeSearch(btn.prompt)}
                  className="flex items-center gap-2 bg-[#f4f7fa] text-gray-600 px-3.5 py-2 rounded-full text-[12px] font-semibold whitespace-nowrap hover:bg-gray-100 hover:text-gray-900 transition-colors border border-gray-100"
                >
                  <btn.icon className="w-3.5 h-3.5 opacity-70" strokeWidth={2.5} />
                  {btn.text}
                </button>
              ))}
            </div>

          </div>
        </div>

      </div>
    </div>
  );
}
