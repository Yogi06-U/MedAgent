"""Phase 1 Playwright End-to-End browser test verifying live UI stepper and case flow."""

import os
import subprocess
import time
import pytest
from playwright.sync_api import sync_playwright

def test_phase1_ui_flow():
    # Start FastAPI server via uvicorn in background
    proc = subprocess.Popen(
        [
            os.path.abspath(".venv/bin/uvicorn"),
            "medagent.api:app",
            "--host", "127.0.0.1",
            "--port", "8008",
        ],
        cwd=os.path.abspath("."),
        env=dict(os.environ, PYTHONPATH=f"{os.path.abspath('backend')}:{os.environ.get('PYTHONPATH', '')}")
    )

    try:
        # Wait for server ready
        time.sleep(2.5)

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1440, "height": 900})
            page = context.new_page()

            # Visit UI
            page.goto("http://127.0.0.1:8008", wait_until="networkidle")
            page.wait_for_selector("text=MedAgent", timeout=8000)

            # Click synthetic case 1
            case_btn = page.locator("text=1. Community-Acquired Pneumonia")
            case_btn.click()
            time.sleep(0.5)

            # Click Analyse Case
            analyse_btn = page.locator("button:has-text('Analyse Case')")
            analyse_btn.click()

            # Wait for Stepper to show done on plan
            page.wait_for_selector("text=Orchestrator Plan", timeout=8000)

            # Wait for physician gate bar to show Approve
            page.wait_for_selector("text=Approve & Release", timeout=8000)
            time.sleep(1)

            # Capture Phase 1 Stepper proof screenshot
            os.makedirs("screenshots", exist_ok=True)
            page.screenshot(path="screenshots/phase1_stepper.png", full_page=True)
            print("Captured screenshot: screenshots/phase1_stepper.png")

            # Click Approve & Release
            approve_btn = page.locator("button:has-text('Approve & Release')")
            approve_btn.click()

            # Wait for PDF Download button
            page.wait_for_selector("text=Download Report PDF", timeout=8000)
            time.sleep(0.5)

            # Capture Phase 1 Approved proof screenshot
            page.screenshot(path="screenshots/phase1_approved.png", full_page=True)
            print("Captured screenshot: screenshots/phase1_approved.png")

            browser.close()
    finally:
        proc.terminate()
        proc.wait(timeout=5)
