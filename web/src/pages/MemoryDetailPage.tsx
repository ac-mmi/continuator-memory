import { Link, useParams } from "react-router-dom";
import { MemoryDetail } from "../components/memory/MemoryDetail";
import { TopBar } from "../components/shell/TopBar";
import { EmptyState, ErrorNotice, LoadingRows } from "../components/ui/EmptyState";
import { Icon } from "../components/ui/Icon";
import { useService } from "../data/ServiceProvider";
import { useAsync } from "../lib/useAsync";
import styles from "./pages.module.css";

export function MemoryDetailPage() {
  const { memoryId = "" } = useParams();
  const service = useService();
  const detail = useAsync(() => service.getMemory(memoryId), [service, memoryId]);

  return (
    <>
      <TopBar
        leading={
          <Link to="/memories" className={styles.backLink} aria-label="Back to memories">
            <Icon name="arrowLeft" size={15} />
          </Link>
        }
        title={
          <span className={styles.crumbs}>
            <Link to="/memories">Memories</Link>
            <Icon name="chevronRight" size={13} />
            <span>{detail.status === "ready" && detail.data ? detail.data.memory.title : "Memory"}</span>
          </span>
        }
      />
      <div className={styles.scroll}>
        {detail.status === "loading" && (
          <div className={styles.page}>
            <LoadingRows rows={5} />
          </div>
        )}
        {detail.status === "error" && (
          <div className={styles.page}>
            <ErrorNotice error={detail.error} onRetry={detail.reload} />
          </div>
        )}
        {detail.status === "ready" && !detail.data && (
          <EmptyState title="Memory not found" description="It may have been removed or merged into another memory." />
        )}
        {detail.status === "ready" && detail.data && <MemoryDetail detail={detail.data} />}
      </div>
    </>
  );
}
