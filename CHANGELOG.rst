Changelog
=========

Version 3.1 (2024-03-21)
------------------------

Bug Fixes
---------
  - Fix a bug where all data_vars for a task returning dicts
    were converted to objects if any of them needed to be converted.

Version 3.0 (2024-03-16)
--------------------------

Features
--------
  - Major improvement on execution through lazy evaluation of
    settings with relations. When sweeping, only settings which
    dependencies have changed are evaluated.

Bug Fixes
---------
  - Fix an error where sweeping a setting with relations
    prevented result dataset from being created.
