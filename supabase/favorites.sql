-- Design Finder: per-user favourite designs.
-- Run once in Supabase: Dashboard -> SQL Editor -> New query -> paste -> Run.
-- Safe to run again.

create table if not exists public.favorites (
  user_id    uuid not null references auth.users (id) on delete cascade,
  design_uid integer not null,
  design_id  text not null,
  created_at timestamptz not null default now(),
  primary key (user_id, design_uid)
);

create index if not exists favorites_user_time on public.favorites (user_id, created_at desc);

-- No browser access; only the Design Finder server (secret key) reads and writes.
alter table public.favorites enable row level security;
revoke all on public.favorites from anon, authenticated;
