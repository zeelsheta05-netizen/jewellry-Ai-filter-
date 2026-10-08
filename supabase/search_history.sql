-- Design Finder: per-user search history (query, what the AI understood, results).
-- Run once in Supabase: Dashboard -> SQL Editor -> New query -> paste -> Run.
-- Safe to run again.

create table if not exists public.search_history (
  id                bigint generated always as identity primary key,
  user_id           uuid not null references auth.users (id) on delete cascade,
  query             text not null check (char_length(query) <= 500),
  category_override text,
  metal_override    text,
  understood        jsonb not null default '{}',
  notes             jsonb not null default '[]',
  matches           integer not null default 0,
  result_uids       integer[] not null default '{}',
  created_at        timestamptz not null default now()
);

create index if not exists search_history_user_time
  on public.search_history (user_id, created_at desc);

-- Same lock-down as profiles: no browser access at all. Only the Design Finder
-- server (secret key) reads and writes, always filtered to the signed-in user.
alter table public.search_history enable row level security;
revoke all on public.search_history from anon, authenticated;
