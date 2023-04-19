#!/bin/bash

poetry version patch
VERSION=$(poetry version --short)
git add pyproject.toml
git commit -m "Bump to $VERSION"
git tag "$VERSION"
git push --tags
