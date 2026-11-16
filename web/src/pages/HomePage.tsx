import { Link, useParams } from "react-router-dom";
import { ChatView } from "../components/chat/ChatView";
import { TopBar } from "../components/shell/TopBar";
import { EmptyState } from "../components/ui/EmptyState";
import styles from "./pages.module.css";

export function HomePage() {
  const { conversationId } = useParams();
  return <ChatView key={conversationId ?? "new"} conversationId={conversationId} />;
}

export function NotFoundPage() {
  return (
    <>
      <TopBar title="Not found" />
      <EmptyState
        title="This page doesn't exist"
        action={
          <Link to="/" className={styles.textLink}>
            Go home
          </Link>
        }
      />
    </>
  );
}
