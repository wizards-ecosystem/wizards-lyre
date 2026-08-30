import { ProjectSummary } from "../api";
import { Icon } from "./Icon";
import { ProjectRail } from "./Workbench";

/**
 * The project library: search, create, open, favorite, delete, and the
 * jukebox-style inline preview.
 *
 * Every prop below is state that only the library uses. Pulling it out of App
 * is what makes that true rather than merely likely -- nothing else in the app
 * can reach `creatingProject` or `previewProjectId` any more.
 */
export function LibraryPane({
  open,
  onClose,
  projects,
  filteredProjects,
  librarySearch,
  onSearchChange,
  activeId,
  onOpenProject,
  onToggleFavorite,
  onDeleteProject,
  creatingProject,
  onStartCreating,
  onCancelCreating,
  onCreateProject,
  newTitle,
  onNewTitleChange,
  newQuery,
  onNewQueryChange,
  previewProjectId,
  onTogglePreview,
  previewAudioRef,
  onPreviewEnded,
}: {
  open: boolean;
  onClose: () => void;
  projects: ProjectSummary[];
  filteredProjects: ProjectSummary[];
  librarySearch: string;
  onSearchChange: (value: string) => void;
  activeId: string | null;
  onOpenProject: (id: string) => void;
  onToggleFavorite: (project: ProjectSummary) => void;
  onDeleteProject: (project: ProjectSummary) => void;
  creatingProject: boolean;
  onStartCreating: () => void;
  onCancelCreating: () => void;
  onCreateProject: () => void;
  newTitle: string;
  onNewTitleChange: (value: string) => void;
  newQuery: string;
  onNewQueryChange: (value: string) => void;
  previewProjectId: string | null;
  onTogglePreview: (project: ProjectSummary) => void;
  previewAudioRef: React.MutableRefObject<HTMLAudioElement | null>;
  onPreviewEnded: () => void;
}) {
  return (
    <ProjectRail open={open}>
      <div className="rail-heading">
        <div>
          <span className="eyebrow">Projects</span>
          <h2>Library</h2>
        </div>
        <button
          type="button"
          className="icon-button drawer-close"
          aria-label="Close projects"
          onClick={onClose}
        >
          <Icon name="close" />
        </button>
      </div>

      {!creatingProject ? (
        <button type="button" className="new-project-trigger" onClick={onStartCreating}>
          <Icon name="add" />
          New project
        </button>
      ) : (
        <form
          className="new-project"
          onSubmit={(event) => {
            event.preventDefault();
            onCreateProject();
          }}
          onKeyDown={(event) => {
            if (event.key === "Escape") onCancelCreating();
          }}
        >
          <div className="composer-heading">
            <span>New composition</span>
            <button
              type="button"
              className="icon-button"
              aria-label="Cancel new project"
              onClick={onCancelCreating}
            >
              <Icon name="close" />
            </button>
          </div>
          <label>
            Title
            <input
              autoFocus
              placeholder="title"
              value={newTitle}
              onChange={(event) => onNewTitleChange(event.target.value)}
            />
          </label>
          <label>
            Starting idea <span className="optional">optional</span>
            <textarea
              placeholder="simple query (optional)"
              value={newQuery}
              onChange={(event) => onNewQueryChange(event.target.value)}
            />
          </label>
          <button type="submit" className="button-primary">
            Create project
          </button>
        </form>
      )}

      <label className="search-field">
        <Icon name="search" />
        <span className="sr-only">Search projects</span>
        <input
          className="library-search"
          placeholder="Search projects"
          value={librarySearch}
          onChange={(event) => onSearchChange(event.target.value)}
        />
      </label>

      <ul className="project-list">
        {filteredProjects.map((project) => (
          <li key={project.id} className={project.id === activeId ? "active" : ""}>
            <button
              type="button"
              className="project-row"
              aria-label={`Open ${project.title}`}
              onClick={() => {
                onOpenProject(project.id);
                onClose();
              }}
            >
              <span className="project-title">{project.title}</span>
              <span className="project-updated">
                {project.updated_at ? new Date(project.updated_at).toLocaleDateString() : ""}
              </span>
            </button>
            <div className="project-actions">
              <button
                type="button"
                className={`icon-button favorite-btn ${project.favorite ? "favorited" : ""}`}
                title={project.favorite ? "Unfavorite" : "Favorite"}
                aria-label={`${project.favorite ? "Unfavorite" : "Favorite"} ${project.title}`}
                onClick={() => onToggleFavorite(project)}
              >
                <Icon name="star" />
              </button>
              <button
                type="button"
                className="icon-button preview-btn"
                title={
                  project.active_take_id
                    ? previewProjectId === project.id
                      ? "Pause preview"
                      : "Play last take"
                    : "No takes yet"
                }
                aria-label={`${previewProjectId === project.id ? "Pause" : "Play"} ${project.title} preview`}
                disabled={!project.active_take_id}
                onClick={() => onTogglePreview(project)}
              >
                <Icon name={previewProjectId === project.id ? "pause" : "play"} />
              </button>
              <button
                type="button"
                className="icon-button delete-btn"
                title="Delete project"
                aria-label={`Delete ${project.title}`}
                onClick={() => onDeleteProject(project)}
              >
                <Icon name="delete" />
              </button>
            </div>
          </li>
        ))}
      </ul>
      {filteredProjects.length === 0 && (
        <p className="empty-copy">
          {projects.length === 0 ? "No projects yet." : "No projects match this search."}
        </p>
      )}
      <audio ref={previewAudioRef} onEnded={onPreviewEnded} hidden />
    </ProjectRail>
  );
}
