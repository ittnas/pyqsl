Changelog
=========

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
