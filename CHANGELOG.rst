Changelog
=========

Version 3.2 (2024-03-21)
------------------------

Features
--------
  - Add a new option to run, use_shallow_copy to enable the use of
     shallow copies instead of deep copies.

Bug Fixes
---------
  - When creating dataset, data vars are converted one by one to
    prevent casting all of them if a single conversion results in
    exception.
  - Fix an issue where setting resolution with sweeps converted array-
    like setting values to object arrays.

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
