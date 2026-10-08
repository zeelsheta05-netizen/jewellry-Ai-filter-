-- Design Finder: admin role (for the upload panel at /admin).
-- Run once in Supabase: Dashboard -> SQL Editor -> New query -> paste -> Run.
-- Then tick `is_admin` for your own row in Table Editor -> profiles.
-- Takes effect within a minute; untick to remove admin rights.

alter table public.profiles add column if not exists is_admin boolean not null default false;
