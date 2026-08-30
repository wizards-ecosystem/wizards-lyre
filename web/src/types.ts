// Shared UI types. Domain types (Plan, Take, Job, ...) live in api.ts.

export type SaveState = "idle" | "saving" | "saved" | "error";
export type InspectorTab = "takes" | "styles";
export type OperationGroup = "create" | "transform" | "tracks";
export type IconName =
  | "add"
  | "back"
  | "close"
  | "delete"
  | "library"
  | "pause"
  | "play"
  | "search"
  | "settings"
  | "spark"
  | "star"
  | "wave";

export interface ConfirmationRequest {
  title: string;
  message: string;
  confirmLabel: string;
  destructive?: boolean;
  resolve: (accepted: boolean) => void;
}
