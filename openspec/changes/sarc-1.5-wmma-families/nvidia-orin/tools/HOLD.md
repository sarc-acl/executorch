# Coordinator hold

The detached build queue checks `<artifacts>/HOLD` before source export and before
starting a build, including after acquiring the shared desktop build lock.
While held its JSON status is `queued`; it checks again every 30 seconds.
The coordinator owns the file. The queue never creates or removes it.

This queue performs no target deployment or GPU execution. Future target queues
must also check the deployment root's `HOLD` at every GPU job boundary.
