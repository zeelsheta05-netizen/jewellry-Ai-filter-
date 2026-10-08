-- Design Finder: the collection of designs scraped from other jewellers' websites.
-- Run once in Supabase: Dashboard -> SQL Editor -> New query -> paste -> Run.
-- Safe to run again. Needs brand_designs.sql first.
--
-- Every product page the team fetches at /brand-import gets a row here, listed
-- or not. Its files (pictures at full size, the page, design.json) are in the
-- dataset storage under `folder` ("Web Designs/<site>/<date>_<name>_<id>").
-- archive_status: 'staged' while they wait on the Design Finder Mac,
-- 'stored' once they are in the dataset storage.

create table if not exists public.web_designs (
  id               bigint generated always as identity primary key,
  url              text not null check (url ~* '^https?://'),
  site             text not null,
  title            text not null default '',
  brand            text not null default '',
  price            numeric(14, 2),
  currency         text not null default '',
  original         jsonb not null,
  pictures         jsonb not null default '[]',
  folder           text not null unique,
  archive_status   text not null default 'staged' check (archive_status in ('staged', 'stored')),
  brand_design_id  bigint references public.brand_designs (id) on delete set null,
  scraped_by       uuid references auth.users (id) on delete set null,
  scraped_by_name  text not null default '',
  created_at       timestamptz not null default now(),
  updated_at       timestamptz not null default now()
);

create index if not exists web_designs_time on public.web_designs (created_at desc);
create index if not exists web_designs_site on public.web_designs (site, created_at desc);

-- No browser access; only the Design Finder server (secret key) reads and writes.
alter table public.web_designs enable row level security;
revoke all on public.web_designs from anon, authenticated;
