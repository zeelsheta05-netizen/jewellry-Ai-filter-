-- Design Finder: orders and the jeweler role (the jeweler panel at /jeweler).
-- Run once in Supabase: Dashboard -> SQL Editor -> New query -> paste -> Run.
-- Safe to run again.
--
-- Jewelers are the manufacturing side. They sign up like everyone else, an
-- admin approves them, and then ticks `is_jeweler` for their row in
-- Table Editor -> profiles. Takes effect within a minute; untick to remove.
-- Admins can open the jeweler panel too.

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
