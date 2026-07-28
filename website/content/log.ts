export type LogEntry = {
  id: string;
  date: string;
  title: string;
  description: string;
  href?: string;
};

export const log: LogEntry[] = [];
