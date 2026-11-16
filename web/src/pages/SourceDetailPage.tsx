import { Link, useParams } from "react-router-dom";
import { TopBar } from "../components/shell/TopBar";
import { SourceDetail } from "../components/source/SourceDetail";
import { EmptyState, ErrorNotice, LoadingRows } from "../components/ui/EmptyState";
import { Icon } from "../components/ui/Icon";
import { useService } from "../data/ServiceProvider";
import { useAsync } from "../lib/useAsync";
import styles from "./pages.module.css";

export function SourceDetailPage() {
  const { sourceId = "" } = useParams();
  const service = useService();
  const detail = useAsync(() => service.getSource(sourceId), [service, sourceId]);

  return (
    <>
      <TopBar
        leading={
          <Link to="/sources" className={styles.backLink} aria-label="Back to sources">
            <Icon name="arrowLeft" size={15} />
          </Link>
        }
        title={
          <span className={styles.crumbs}>
            <Link to="/sources">Sources</Link>
            <Icon name="chevronRight" size={13} />
            <span>{detail.status === "ready" && detail.data ? detail.data.source.title : "Source"}</span>
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
          <EmptyState title="Source not found" description="It may have been removed." />
        )}
        {detail.status === "ready" && detail.data && <SourceDetail detail={detail.data} />}
      </div>
    </>
  );
}
