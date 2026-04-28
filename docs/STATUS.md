# 🧠 BCNN Accelerator — Phase B Status Report

## 1. Overview

This document provides a **structured evaluation of the RTL implementation (Phase B)** of the BCNN Edge Vision Accelerator, aligned with the requirements defined in `SPEC.md`.

The focus is on:

- Architectural correctness
- Hardware design quality
- FPGA readiness (timing, memory, scalability)
- Verification completeness

> ⚠ Phase A (ML training, binarization pipeline, .mif generation) is intentionally excluded from this report.

---

## 2. System Architecture Summary

The implemented system follows a **fully streaming dataflow pipeline**:

```
SPI → FIFO → BIN → LB1 → BCNN_L1 → LB2 → BCNN_L2 → POOL → CLASSIFIER
         ↑                                                     ↓
         └────────────── FSM CONTROLLER ───────────────────────┘
```

### Key Characteristics

- **Streaming execution** (no frame-level buffering)
- **Dataflow-driven pipeline** (compute triggered by valid signals)
- **Modular RTL design**
- **No multipliers** (XNOR + popcount only)

---

## 3. Phase B Spec Compliance Summary

| Component             | Requirement (SPEC.md)               | Status | Notes                          |
| --------------------- | ----------------------------------- | ------ | ------------------------------ |
| SPI Interface         | 4-wire, 20MHz, CDC-safe             | ✅     | Fully compliant                |
| FIFO                  | CDC buffering                       | ✅     | Single-clock valid alternative |
| Binarization          | ±1 mapping                          | ⚠️     | Hardcoded threshold            |
| XNOR + Popcount       | No multipliers, efficient reduction | ⚠️     | Not fully Fmax-optimized       |
| BCNN Layer (L1)       | Parameterized                       | ⚠️     | Single-channel assumption      |
| BCNN Layer (L2)       | Multi-channel accumulation          | ⚠️     | Timing-heavy                   |
| Line Buffer           | Sliding window                      | ✅     | Correct implementation         |
| FSM Controller        | Orchestrate pipeline                | ✅     | L2-aware and correct           |
| Memory (BRAM/M4K)     | Use internal block RAM              | ⚠️     | Not guaranteed                 |
| Top-Level Integration | Full pipeline                       | ✅     | Clean streaming design         |
| Testbench             | Simulation + latency measurement    | ⚠️     | Partial validation             |
| Multipliers           | Forbidden                           | ✅     | Fully compliant                |
| Synchronous Design    | Required                            | ✅     | Fully compliant                |

### Overall Assessment

> ✔ **Functionally complete RTL pipeline**
> ❗ **Not yet timing-clean or benchmark-complete**

---

## 4. Module-Level Evaluation

---

### 4.1 SPI Interface + FIFO

**What works**

- Proper 2-stage synchronization of SPI signals
- Edge-based SCK detection
- Clean byte-level interface (`byte_valid`)
- FIFO decouples ingestion from compute

**Verdict**
✔ Fully compliant and production-quality

---

### 4.2 Binarization Stage

**Implementation**

```verilog
pixel_bin = (fifo_out > 127);
```

**Assessment**

- Functionally correct
- Hardcoded threshold

**Issue**

- Not aligned with dataset-driven binarization (training pipeline)

**Verdict**
⚠ Acceptable for RTL, but not ML-consistent

---

### 4.3 Line Buffer (LB1 & LB2)

**What works**

- Correct 3×3 sliding window generation
- Row buffering using BRAM-style arrays
- Multi-channel support in LB2
- Proper valid signal generation after warm-up

**Verdict**
✔ Fully correct and spec-compliant

---

### 4.4 BCNN Layer 1

**What works**

- XNOR-based multiplication
- Popcount accumulation
- Threshold-based activation
- Parameterized kernel size

**Limitations**

- Assumes single input channel
- No cross-channel accumulation

**Verdict**
✔ Functionally correct
⚠ Architecturally limited

---

### 4.5 BCNN Layer 2

**What works**

- Multi-channel convolution
- Per-channel popcount
- Cross-channel accumulation
- Threshold activation

**Critical Issue**

- Fully combinational accumulation:

```verilog
sum[i] = sum[i] + pc[i][j];
```

**Impact**

- Long critical path:

  ```
  XNOR → popcount → multi-input add → compare
  ```

- High risk of timing failure on Cyclone IV

**Verdict**
✔ Functionally correct
❗ Major timing risk

---

### 4.6 Popcount Module

**What works**

- XNOR stage implemented correctly
- Multi-level reduction tree

**Issue**

- Final accumulation uses loop-based reduction (not fully balanced)
- No pipelining

**Verdict**
✔ Correct
⚠ Not optimized for high Fmax

---

### 4.7 Pooling + Classifier

**What works**

- Global accumulation per channel
- Argmax classification
- Correct avoidance of division (mean ≡ sum for argmax)

**Issue**

- Argmax implemented as large combinational block

**Verdict**
✔ Correct
⚠ Timing risk at scale

---

### 4.8 FSM Controller

**What works**

- Clean separation of control and datapath
- Tracks L2 outputs (correct abstraction)
- Proper frame lifecycle management

**Verdict**
✔ Fully compliant and well-designed

---

### 4.9 Top-Level Integration

**What works**

- Clean streaming datapath
- Continuous FIFO consumption
- Proper module composition

**Limitation**

- No backpressure or stall handling

**Verdict**
✔ Architecturally correct

---

### 4.10 Testbench

**What works**

- SPI stimulus generation
- `$readmemh` usage
- Latency measurement

**Missing**

- No correctness validation vs ground truth
- No multi-test coverage
- Idealized SPI timing (no stress testing)

**Verdict**
⚠ Partial validation only

---

## 5. Timing & Synthesizability Assessment

### Critical Paths

1. **BCNN Layer 2 accumulation**
2. **Popcount final reduction**
3. **Argmax classifier**

### Expected Issues

- Reduced Fmax due to deep combinational paths
- Potential failure to meet timing on Cyclone IV
- No pipelining to break critical paths

### Conclusion

> ❗ The design is **functionally correct but not timing-clean**

---

## 6. Memory (BRAM) Compliance

### Current Approach

- Use of:

```verilog
(* ramstyle = "M4K" *)
```

### Issue

- This is a **hint**, not a guarantee
- No explicit dual-port RAM usage
- No Quartus verification of inference

### Conclusion

> ⚠ BRAM usage is **likely but not guaranteed**

---

## 7. Resource & Scalability Considerations

### Observations

- Fully parallel architecture:
  - All filters active simultaneously
  - All channels computed in parallel

### Implications

- High resource usage
- Poor scalability with:
  - increasing channels
  - larger kernels

### Conclusion

> ⚠ Design favors correctness over scalability

---

## 8. Known Risks

- ❗ Timing closure failure (critical)
- ⚠ BRAM inference uncertainty
- ⚠ High combinational fan-in (L2 + classifier)
- ⚠ No pipelining
- ⚠ No backpressure handling

---

## 9. What Is Verified vs Not Verified

### Verified

- End-to-end dataflow
- SPI ingestion
- Window generation
- Layer execution
- Latency measurement

### Not Verified

- Output correctness vs reference model
- Multi-image robustness
- FIFO stress conditions
- Timing behavior under realistic load

---

## 10. Final Project Snapshot

| Area              | Status | Notes               |
| ----------------- | ------ | ------------------- |
| SPI interface     | ✅     | Clean, synchronized |
| FIFO              | ✅     | Correct, safe       |
| Binarization      | ⚠️     | Hardcoded threshold |
| Line Buffer L1    | ✅     | Correct             |
| BCNN L1           | ⚠️     | Single-channel      |
| Line Buffer L2    | ✅     | Correct             |
| BCNN L2           | ⚠️     | Timing-heavy        |
| Popcount          | ⚠️     | Not fully optimized |
| Pool + Classifier | ⚠️     | Timing risk         |
| FSM               | ✅     | Correct             |
| Top module        | ✅     | Clean streaming     |
| Testbench         | ⚠️     | Partial             |
| BRAM usage        | ⚠️     | Not guaranteed      |
| Timing readiness  | ❌     | Not achieved        |

---

## 11. Final Verdict

The current system represents:

> ✔ A **functionally correct, end-to-end BCNN accelerator**
> ❗ But **not yet an FPGA-ready, timing-clean implementation**

### To reach full Phase B completion:

- Add pipelining (popcount + L2 accumulation)
- Verify BRAM inference in Quartus
- Improve testbench with correctness validation
- Reduce critical path depth

---

## 12. One-Line Summary

> A **correct streaming BCNN accelerator**, but not yet a **timing-optimized or fully validated FPGA design**.
