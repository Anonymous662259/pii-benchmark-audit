#!/usr/bin/env bash
# Build acl2023_pii_audit.pdf out of tree (no .aux/.log clutter in this folder).
#   bash paper/build_pii_audit.sh            # from the repo root
# Run scripts/rare_pii_tables.py and scripts/audit_paper_tables.py first so generated/ is current.
# Five passes: the ACL review line numbers (lineno) need them to settle in the margins.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
build="$(mktemp -d)"
cp -r "$here/acl2023.sty" "$here/acl_natbib.bst" "$here/acl2027.bib" "$here/rare_pii_extra.bib" \
      "$here/pii_audit_extra.bib" "$here/acl2023_pii_audit.tex" "$here/figs" "$here/generated" "$build/"
cd "$build"
pdflatex -interaction=nonstopmode acl2023_pii_audit.tex > /dev/null || true
bibtex acl2023_pii_audit > /dev/null || true
for _ in 1 2 3; do pdflatex -interaction=nonstopmode acl2023_pii_audit.tex > /dev/null || true; done
if grep -q "^!" acl2023_pii_audit.log; then
  grep -A3 "^!" acl2023_pii_audit.log | head -20
  echo "BUILD FAILED (log: $build/acl2023_pii_audit.log)"; exit 1
fi
grep -E "Citation .* undefined|Reference .* undefined|Overfull" acl2023_pii_audit.log || true
grep -iE "^Warning--|error" acl2023_pii_audit.blg || true
cp acl2023_pii_audit.pdf "$here/acl2023_pii_audit.pdf"
echo "OK -> $here/acl2023_pii_audit.pdf  (build dir: $build)"
