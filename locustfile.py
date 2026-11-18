"""Load test:  locust -f locustfile.py --host http://localhost:8000

Mixes cached and uncached questions so you can report throughput and the
latency difference caching makes (see GET /usage for server-side numbers).
"""
import random
import time
import uuid

from locust import HttpUser, between, task

SAMPLE = (
    "Refund policy. Customers may request a refund within 30 days of purchase. "
    "Shipping takes 4 days from the Hyderabad warehouse. Support is open 9 to 5."
)
QUESTIONS = [
    "How many days do customers have to request a refund?",
    "Where do orders ship from?",
    "When is support open?",
    "What is the delivery time?",
]


class DocQAUser(HttpUser):
    wait_time = between(0.5, 2)

    def on_start(self):
        email = f"load-{uuid.uuid4().hex[:10]}@example.com"
        self.client.post("/auth/register", json={"email": email, "password": "load-test-pass1"})
        token = self.client.post("/auth/login", data={"username": email, "password": "load-test-pass1"}).json()[
            "access_token"
        ]
        self.client.headers["Authorization"] = f"Bearer {token}"
        doc = self.client.post("/documents", files={"file": ("sample.txt", SAMPLE.encode())}).json()
        for _ in range(60):  # wait for the worker to finish ingestion
            if self.client.get(f"/documents/{doc['id']}", name="/documents/[id]").json()["status"] == "ready":
                break
            time.sleep(1)

    @task(8)
    def ask_repeated(self):  # mostly cache hits
        self.client.post("/ask", json={"question": random.choice(QUESTIONS)}, name="/ask (cacheable)")

    @task(2)
    def ask_uncached(self):
        self.client.post(
            "/ask", json={"question": random.choice(QUESTIONS), "no_cache": True}, name="/ask (no cache)"
        )

    @task(1)
    def list_documents(self):
        self.client.get("/documents")
