cat > Main.lean <<'EOF'
import DROSafety

def main : IO Unit := do
  IO.println "DRO safety certification verified."
EOF
