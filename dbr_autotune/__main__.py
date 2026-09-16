from .cli import main

# The guard is required, not cosmetic: decoding runs in child processes, and on
# Windows those are spawned by re-importing this module. Without it every worker
# would start its own copy of the whole command.
if __name__ == "__main__":
    raise SystemExit(main())
