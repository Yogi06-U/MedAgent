"""Phase 4 Playwright test suite running all 3 synthetic clinical cases and capturing proof screenshots."""

import os
import subprocess
import time
import pytest
from playwright.sync_api import sync_playwright


def test_ui_three_clinical_cases():
    proc = subprocess.Popen(
        [
            os.path.abspath(".venv/bin/uvicorn"),
            "medagent.api:app",
            "--host", "127.0.0.1",
            "--port", "8009",
        ],
        cwd=os.path.abspath("."),
        env=dict(os.environ, PYTHONPATH=f"{os.path.abspath('backend')}:{os.environ.get('PYTHONPATH', '')}")
    )

    try:
        time.sleep(2.5)
        os.makedirs("screenshots", exist_ok=True)

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1440, "height": 900})
            page = context.new_page()

            page.goto("http://127.0.0.1:8009", wait_until="networkidle")
            page.wait_for_selector("text=MedAgent", timeout=8000)

            # -------------------------------------------------------------
            # CASE 1: Infective Community-Acquired Pneumonia with CXR
            # -------------------------------------------------------------
            print("Running Case 1: Community-Acquired Pneumonia...")
            page.locator("text=1. Community-Acquired Pneumonia").click()
            time.sleep(0.5)

            # Upload synthetic CXR
            sample_cxr = os.path.abspath("samples/sample_cxr.png")
            page.set_input_files('input[type="file"][accept*=".png"]', sample_cxr)
            time.sleep(0.5)

            # Run Analysis
            page.locator("button:has-text('Analyse Case')").click()

            # Wait for pipeline to complete reasoning and reach physician review
            page.wait_for_selector("text=Attending Physician:", timeout=25000)
            time.sleep(1.5)

            # Switch to Imaging Tab to inspect CXR + Grad-CAM overlay
            imaging_tab = page.locator("button:has-text('Imaging (CXR)')")
            if imaging_tab.count() > 0:
                imaging_tab.click()
                time.sleep(1.0)

            page.screenshot(path="screenshots/phase4_case1_infective.png", full_page=True)
            print("Captured: screenshots/phase4_case1_infective.png")

            # Test Evidence Drawer
            summary_tab = page.locator("button:has-text('Summary & Differential')")
            summary_tab.click()
            time.sleep(0.5)

            citation_btn = page.locator("button:has-text('S1')").first
            if citation_btn.count() > 0:
                citation_btn.click()
                time.sleep(0.8)
                page.screenshot(path="screenshots/phase4_evidence_drawer.png", full_page=True)
                print("Captured: screenshots/phase4_evidence_drawer.png")
                # Close drawer
                close_drawer = page.locator("button:has(svg.lucide-x)").first
                if close_drawer.count() > 0:
                    close_drawer.click()

            # Approve Case 1
            approve_btn = page.locator("button:has-text('Approve & Release')")
            if approve_btn.count() > 0:
                approve_btn.click()
                time.sleep(1.0)

            # -------------------------------------------------------------
            # CASE 2: Urgent AKI & Hyperkalemia (Blocked Safety Guardrail)
            # -------------------------------------------------------------
            print("Running Case 2: AKI & Severe Hyperkalemia...")
            page.locator("text=2. AKI & Severe Hyperkalemia").click()
            time.sleep(0.5)

            page.locator("button:has-text('Analyse Case')").click()
            page.wait_for_selector("text=Draft BLOCKED by Safety Agent", timeout=25000)
            time.sleep(1.5)

            # Verify Direct Approve button is NOT rendered
            approve_btns = page.locator("button:has-text('Approve & Release')")
            assert approve_btns.count() == 0, "Approve button must NOT be rendered when draft is blocked!"

            page.screenshot(path="screenshots/phase4_case2_urgent_blocked.png", full_page=True)
            print("Captured: screenshots/phase4_case2_urgent_blocked.png")

            # Test Edit Summary Modal
            edit_btn = page.locator("button:has-text('Edit Summary')")
            edit_btn.click()
            time.sleep(0.8)

            page.screenshot(path="screenshots/phase4_edit_modal.png", full_page=True)
            print("Captured: screenshots/phase4_edit_modal.png")

            # Save and release with edits
            page.locator("button:has-text('Save & Release with Edits')").click()
            time.sleep(1.0)

            # -------------------------------------------------------------
            # CASE 3: Routine / Normal-ish Iron Deficiency Anemia
            # -------------------------------------------------------------
            print("Running Case 3: Microcytic Iron Deficiency Anemia...")
            page.locator("text=3. Microcytic Iron Deficiency Anemia").click()
            time.sleep(0.5)

            page.locator("button:has-text('Analyse Case')").click()
            page.wait_for_selector("text=Attending Physician:", timeout=25000)
            time.sleep(1.5)

            # Switch to Labs Tab
            labs_tab = page.locator("button:has-text('Laboratory Results')")
            labs_tab.click()
            time.sleep(1.0)

            page.screenshot(path="screenshots/phase4_case3_routine.png", full_page=True)
            print("Captured: screenshots/phase4_case3_routine.png")

            # -------------------------------------------------------------
            # THEME TEST: Dark Mode
            # -------------------------------------------------------------
            theme_btn = page.locator("button[aria-label*='theme']")
            theme_btn.click()
            time.sleep(0.5)

            page.screenshot(path="screenshots/phase4_dark_mode.png", full_page=True)
            print("Captured: screenshots/phase4_dark_mode.png")

            browser.close()
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            proc.kill()
