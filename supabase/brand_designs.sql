-- Design Finder: designs from other jewellers' websites (the team panel at /brand-import).
-- Run once in Supabase: Dashboard -> SQL Editor -> New query -> paste -> Run.
-- Safe to run again. Needs orders.sql first (it adds the "brand" order kind).
--
-- A jeweler or admin pastes a product link from another jeweller's shop. The
-- server reads the page (name, price, specification, pictures) into
-- `original`, exactly as read, and the team's own version of the design
-- (gold weight per purity, diamonds, size, our pricing) into `ours`.
-- Pictures are files on the Design Finder server (data/brand_designs/),
-- listed in `pictures`.

create table if not exists public.brand_designs (
  id              bigint generated always as identity primary key,
  status          text not null default 'listed' check (status in ('listed', 'hidden')),
  source_url      text not null check (source_url ~* '^https?://'),
  site            text not null,
  category        text not null,
  original        jsonb not null,
  ours            jsonb not null,
  pictures        jsonb not null default '[]',
  created_by      uuid references auth.users (id) on delete set null,
  created_by_name text not null default '',
  created_at      timestamptz not null default now(),
  updated_at      timestamptz not null default now()
);

create index if not exists brand_designs_listed on public.brand_designs (status, category, created_at desc);

-- "Buy with us" on a brand design places a normal order of kind 'brand'.
alter table public.orders drop constraint if exists orders_kind_check;
alter table public.orders add constraint orders_kind_check check (kind in ('as_is', 'custom', 'brand'));

-- No browser access; only the Design Finder server (secret key) reads and writes.
alter table public.brand_designs enable row level security;
revoke all on public.brand_designs from anon, authenticated;
