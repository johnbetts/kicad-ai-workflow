"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { cn } from "@/lib/utils";
import { Send, Bot, User, Sparkles, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";

interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: Date;
}

interface AIChatProps {
  /** Initial system context for the AI (e.g., board requirements, current state) */
  systemContext?: string;
  /** Placeholder text for input */
  placeholder?: string;
  /** Callback when AI suggests a change (e.g., component swap, placement fix) */
  onSuggestion?: (suggestion: string) => void;
  /** Whether to show in compact mode (sidebar) vs full mode */
  compact?: boolean;
  /** Custom welcome message */
  welcomeMessage?: string;
  className?: string;
}

export function AIChat({
  systemContext,
  placeholder = "Ask about your design...",
  onSuggestion,
  compact = false,
  welcomeMessage = "I'm your PCB design assistant. I can help with component selection, placement review, DRC fixes, and manufacturing optimization. What would you like to work on?",
  className,
}: AIChatProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "welcome",
      role: "assistant",
      content: welcomeMessage,
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Auto-resize textarea
  const handleInputChange = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    // Auto-resize: reset height then set to scrollHeight
    e.target.style.height = "auto";
    e.target.style.height = Math.min(e.target.scrollHeight, 120) + "px";
  }, []);

  const handleSubmit = useCallback(async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: "user",
      content: input.trim(),
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    // Reset textarea height
    if (inputRef.current) {
      inputRef.current.style.height = "auto";
    }

    try {
      // Call the AI chat API endpoint
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: userMessage.content,
          context: systemContext,
          history: messages.map(m => ({ role: m.role, content: m.content })),
        }),
      });

      if (response.ok) {
        const data = await response.json();
        const assistantMessage: ChatMessage = {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          content: data.response,
          timestamp: new Date(),
        };
        setMessages(prev => [...prev, assistantMessage]);

        if (data.suggestion && onSuggestion) {
          onSuggestion(data.suggestion);
        }
      } else {
        // Fallback: simulate a helpful response when API is not available
        const fallbackMessage: ChatMessage = {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          content: getSimulatedResponse(userMessage.content),
          timestamp: new Date(),
        };
        setMessages(prev => [...prev, fallbackMessage]);
      }
    } catch {
      // Offline fallback
      const fallbackMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: getSimulatedResponse(userMessage.content),
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, fallbackMessage]);
    } finally {
      setIsLoading(false);
    }
  }, [input, isLoading, messages, systemContext, onSuggestion]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  }, [handleSubmit]);

  return (
    <div className={cn(
      "flex flex-col border border-[var(--border)] rounded-lg overflow-hidden",
      compact ? "h-full" : "h-[500px]",
      className
    )}>
      {/* Header */}
      <div className="flex items-center gap-2 px-4 py-2.5 border-b border-[var(--border)] bg-[var(--card)]">
        <Sparkles className="h-4 w-4 text-[var(--primary)]" />
        <span className="text-sm font-medium">AI Design Assistant</span>
        <span className="text-xs text-[var(--muted-foreground)] ml-auto">
          Powered by Claude
        </span>
      </div>

      {/* Messages area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-[var(--background)]">
        {messages.map((msg) => (
          <div key={msg.id} className={cn(
            "flex gap-3",
            msg.role === "user" ? "flex-row-reverse" : "flex-row"
          )}>
            {/* Avatar */}
            <div className={cn(
              "flex-shrink-0 w-7 h-7 rounded-full flex items-center justify-center",
              msg.role === "user"
                ? "bg-[var(--primary)] text-white"
                : "bg-[var(--muted)] text-[var(--foreground)]"
            )}>
              {msg.role === "user" ? <User className="h-3.5 w-3.5" /> : <Bot className="h-3.5 w-3.5" />}
            </div>

            {/* Message bubble */}
            <div className={cn(
              "max-w-[80%] rounded-lg px-3 py-2 text-sm",
              msg.role === "user"
                ? "bg-[var(--primary)] text-white"
                : "bg-[var(--card)] border border-[var(--border)]"
            )}>
              <p className="whitespace-pre-wrap">{msg.content}</p>
              <span className={cn(
                "text-[10px] mt-1 block",
                msg.role === "user" ? "text-blue-200" : "text-[var(--muted-foreground)]"
              )}>
                {msg.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
              </span>
            </div>
          </div>
        ))}

        {/* Loading indicator */}
        {isLoading && (
          <div className="flex gap-3">
            <div className="w-7 h-7 rounded-full bg-[var(--muted)] flex items-center justify-center">
              <Bot className="h-3.5 w-3.5" />
            </div>
            <div className="bg-[var(--card)] border border-[var(--border)] rounded-lg px-3 py-2">
              <div className="flex items-center gap-2 text-sm text-[var(--muted-foreground)]">
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                Thinking...
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input area */}
      <div className="border-t border-[var(--border)] p-3 bg-[var(--card)]">
        <div className="flex gap-2 items-end">
          <textarea
            ref={inputRef}
            value={input}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            rows={1}
            className={cn(
              "flex-1 resize-none rounded-md border border-[var(--border)] bg-[var(--background)]",
              "px-3 py-2 text-sm placeholder:text-[var(--muted-foreground)]",
              "focus:outline-none focus:ring-2 focus:ring-[var(--primary)] focus:border-transparent"
            )}
          />
          <Button
            size="icon"
            onClick={handleSubmit}
            disabled={!input.trim() || isLoading}
          >
            <Send className="h-4 w-4" />
          </Button>
        </div>
        <p className="text-[10px] text-[var(--muted-foreground)] mt-1.5">
          Press Enter to send, Shift+Enter for new line
        </p>
      </div>
    </div>
  );
}

/** Fallback responses when API is not connected */
function getSimulatedResponse(userInput: string): string {
  const lower = userInput.toLowerCase();

  if (lower.includes("component") || lower.includes("part") || lower.includes("capacitor") || lower.includes("resistor")) {
    return "I can help with component selection. For JLCPCB, I recommend using basic parts to avoid the $3 setup fee per extended part. Use the Parts Search to find alternatives. What specific component are you looking for?";
  }
  if (lower.includes("placement") || lower.includes("layout") || lower.includes("move")) {
    return "The placement optimizer uses a 3-level hierarchy: zone partitioning \u2192 group placement \u2192 subcircuit refinement. To improve placement, you can:\n\n1. Click 'Fix It' on any review finding\n2. Lock specific components with fixed constraints\n3. Adjust group priorities in the requirements\n\nWhat aspect of the placement would you like to improve?";
  }
  if (lower.includes("drc") || lower.includes("error") || lower.includes("violation")) {
    return "DRC violations are checked against JLCPCB manufacturing capabilities by default (0.15mm min trace, 0.2mm min drill). Each error has a 'Fix It' button that applies the recommended correction automatically. Would you like me to explain a specific violation?";
  }
  if (lower.includes("cost") || lower.includes("price") || lower.includes("cheap")) {
    return "To optimize cost:\n\n1. **Use basic parts** \u2014 no $3 setup fee each\n2. **2-layer boards** are ~$1.50 cheaper per unit than 4-layer\n3. **Standard 1.6mm FR4** is cheapest\n4. **Combine orders** \u2014 5-board minimum at JLCPCB\n\nYour current BOM uses basic parts where possible. Check the Order tab for a full cost breakdown.";
  }
  if (lower.includes("route") || lower.includes("routing") || lower.includes("trace")) {
    return "Routing is done manually in KiCad after exporting from this tool. The pipeline generates placed but unrouted boards. To export:\n\n1. Download the .kicad_pcb file\n2. Open in KiCad\n3. Route traces (or use the interactive router)\n4. Re-import for validation\n\nWould you like tips on routing strategy?";
  }
  return "I can help with component selection, placement optimization, DRC fixes, cost analysis, and manufacturing preparation. What would you like to work on?\n\nTip: You can also use the 'Fix It' buttons on review findings for one-click corrections.";
}
