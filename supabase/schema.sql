-- Design Finder: user approval table.
-- Run once in Supabase: Dashboard -> SQL Editor -> New query -> paste -> Run.
--
-- Passwords live in Supabase Auth (auth.users, hashed by Supabase).
-- This table adds the admin-approval flag. Tick `approved` in
-- Table Editor -> profiles to let someone in; untick it to lock them out
-- (takes effect within a minute).

create table if not exists public.profiles (
  id            uuid primary key references auth.users (id) on delete cascade,
  email         text not null,
  full_name     text,
  approved      boolean not null default false,
  created_at    timestamptz not null default now(),
  last_login_at timestamptz
);

-- Row level security with no policies: browsers (anon / authenticated keys)
-- can neither read nor write this table. Only the Design Finder server, using
-- the secret key, can.
alter table public.profiles enable row level security;
revoke all on public.profiles from anon, authenticated;

-- Create a profile row (not yet approved) for every new sign-up.
create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = ''
as $$
begin
  insert into public.profiles (id, email, full_name)
  values (new.id, new.email, left(coalesce(new.raw_user_meta_data ->> 'full_name', ''), 80))
  on conflict (id) do nothing;
  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function public.handle_new_user();

-- Search history (also in search_history.sql)
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

-- Favourites (also in favorites.sql)
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

-- Admin role for the upload panel (also in admin.sql)
alter table public.profiles add column if not exists is_admin boolean not null default false;

-- Orders and the jeweler role (also in orders.sql)
alter table public.profiles add column if not exists is_jeweler boolean not null default false;

-- One row per order. What the customer chose and what the buy page showed
-- (job card or CAD figures) is frozen in `snapshot` when the order is placed,
-- so rebuilding the search index or the specs never changes an order.
-- Money is only ever entered by a jeweler (`quote`, `payments`).
create table if not exists public.orders (
  id              bigint generated always as identity primary key,
  user_id         uuid references auth.users (id) on delete set null,
  status          text not null default 'request' check (status in (
                    'request', 'quoted', 'approved', 'advance_paid', 'in_production',
                    'quality_check', 'final_bill', 'ready', 'delivered', 'cancelled')),
  kind            text not null default 'as_is' check (kind in ('as_is', 'custom')),
  design_uid      integer not null,
  design_id       text not null,
  design_key      text not null,            -- design id | folder: stays valid after an index rebuild
  category        text not null,
  metal           text not null,
  purity          text not null,
  ring_size_in    integer,                  -- Indian ring size, rings only
  quantity        integer not null default 1 check (quantity between 1 and 20),
  thumb           text,
  customer_name   text not null,
  customer_email  text not null,
  customer_phone  text not null,
  customer_note   text not null default '' check (char_length(customer_note) <= 1000),
  snapshot        jsonb not null,
  quote           jsonb,
  quoted_total    numeric(14, 2),
  payments        jsonb not null default '[]',
  actual_weight_g numeric(10, 3),
  cancel_reason   text,
  created_at      timestamptz not null default now(),
  updated_at      timestamptz not null default now()
);

create index if not exists orders_status_time on public.orders (status, created_at desc);
create index if not exists orders_user_time on public.orders (user_id, created_at desc);

-- Everything that happens to an order, in order: who did it and when.
create table if not exists public.order_events (
  id          bigint generated always as identity primary key,
  order_id    bigint not null references public.orders (id) on delete cascade,
  actor_id    uuid,
  actor_name  text not null default '',
  actor_role  text not null check (actor_role in ('customer', 'jeweler', 'admin')),
  kind        text not null,
  from_status text,
  to_status   text,
  detail      jsonb not null default '{}',
  created_at  timestamptz not null default now()
);

create index if not exists order_events_order_time on public.order_events (order_id, created_at);

-- No browser access; only the Design Finder server (secret key) reads and writes.
alter table public.orders enable row level security;
alter table public.order_events enable row level security;
revoke all on public.orders from anon, authenticated;
revoke all on public.order_events from anon, authenticated;
