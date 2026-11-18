import io
import json

from reportlab.pdfgen import canvas

from app.core.config import settings
from tests.conftest import upload

REFUND = (
    b"Refund policy. Customers may request a refund within 30 days of purchase. "
    b"Refunds are issued to the original payment method within 5 business days."
)
SHIPPING = b"Shipping information. Orders ship from the Hyderabad warehouse and arrive within 4 days."
UNRELATED = "zebra quantum xylophone"


def ask(client, headers, question, **extra):
    return client.post("/ask", headers=headers, json={"question": question, **extra})


def make_pdf(pages: list[str]) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    for text in pages:
        c.drawString(72, 750, text)
        c.showPage()
    c.save()
    return buf.getvalue()


# ---------- auth ----------
def test_register_login_and_duplicate(client):
    body = {"email": "Dev@Example.com", "password": "longenough1"}
    assert client.post("/auth/register", json=body).status_code == 201
    assert client.post("/auth/register", json=body).status_code == 409
    assert client.post("/auth/register", json={"email": "x@example.com", "password": "short"}).status_code == 422
    ok = client.post("/auth/login", data={"username": "dev@example.com", "password": "longenough1"})
    assert ok.status_code == 200 and {"access_token", "refresh_token"} <= ok.json().keys()
    bad = client.post("/auth/login", data={"username": "dev@example.com", "password": "nope-nope-1"})
    assert bad.status_code == 401
    assert client.post("/auth/login", data={"username": "ghost@example.com", "password": "x" * 9}).status_code == 401


def test_protected_routes_and_token_types(client, alice):
    assert client.get("/documents").status_code == 401
    assert client.get("/documents", headers={"Authorization": "Bearer junk"}).status_code == 401
    creds = {"username": "alice@example.com", "password": "correct-horse-battery"}
    tokens = client.post("/auth/login", data=creds).json()
    refreshed = client.post("/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert refreshed.status_code == 200
    as_access = {"Authorization": f"Bearer {tokens['refresh_token']}"}  # refresh token misused
    assert client.get("/documents", headers=as_access).status_code == 401
    assert client.post("/auth/refresh", json={"refresh_token": tokens["access_token"]}).status_code == 401


# ---------- documents ----------
def test_upload_ingests_and_lists(client, alice):
    r = upload(client, alice, "refunds.txt", REFUND)
    assert r.status_code == 202
    doc = client.get(f"/documents/{r.json()['id']}", headers=alice).json()
    assert doc["status"] == "ready" and doc["chunk_count"] >= 1
    assert [d["filename"] for d in client.get("/documents", headers=alice).json()] == ["refunds.txt"]


def test_upload_validation(client, alice, monkeypatch):
    assert upload(client, alice, "virus.exe", b"MZ").status_code == 415
    assert upload(client, alice, "empty.txt", b"").status_code == 400
    assert upload(client, alice, "a.txt", REFUND).status_code == 202
    assert upload(client, alice, "a-copy.txt", REFUND).status_code == 409  # same content
    monkeypatch.setattr(settings, "max_upload_mb", 1)
    assert upload(client, alice, "big.txt", b"x" * (1024 * 1024 + 1)).status_code == 413
    path_trick = upload(client, alice, "../../etc/passwd.txt", SHIPPING)
    assert path_trick.status_code == 202 and path_trick.json()["filename"] == "passwd.txt"


def test_corrupt_pdf_is_marked_failed(client, alice):
    r = upload(client, alice, "broken.pdf", b"%PDF-1.4 this is not a pdf", "application/pdf")
    doc = client.get(f"/documents/{r.json()['id']}", headers=alice).json()
    assert doc["status"] == "failed" and doc["error"]


def test_upload_rate_limit(client, alice, monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_upload_per_min", 1)
    assert upload(client, alice, "a.txt", REFUND).status_code == 202
    assert upload(client, alice, "b.txt", SHIPPING).status_code == 429


# ---------- asking ----------
def test_answer_has_citations(client, alice):
    upload(client, alice, "refunds.txt", REFUND)
    upload(client, alice, "shipping.txt", SHIPPING)
    r = ask(client, alice, "How many days do customers have to request a refund?")
    assert r.status_code == 200
    body = r.json()
    assert not body["refused"] and "30 days" in body["answer"]
    assert body["citations"] and body["citations"][0]["filename"] == "refunds.txt"
    assert body["retrieved"] and "distance" in body["retrieved"][0]


def test_refuses_when_documents_do_not_contain_answer(client, alice):
    assert ask(client, alice, "What is the refund policy?").json()["refused"]  # no documents at all
    upload(client, alice, "refunds.txt", REFUND)
    body = ask(client, alice, UNRELATED).json()
    assert body["refused"] and body["citations"] == []
    assert body["answer"].startswith("I don't know")


def test_pdf_citations_include_page_numbers(client, alice):
    pdf = make_pdf(["Introduction to our company.", "The warranty period is twenty four months."])
    r = upload(client, alice, "manual.pdf", pdf, "application/pdf")
    assert client.get(f"/documents/{r.json()['id']}", headers=alice).json()["status"] == "ready"
    body = ask(client, alice, "How long is the warranty period?").json()
    assert body["citations"][0]["page"] == 2


def test_modes_and_document_filter(client, alice):
    a = upload(client, alice, "refunds.txt", REFUND).json()["id"]
    upload(client, alice, "shipping.txt", SHIPPING)
    for mode in ("vector", "keyword", "hybrid"):
        body = ask(client, alice, "refund policy", mode=mode, no_cache=True).json()
        assert body["retrieved"], mode
    only_a = ask(client, alice, "warehouse shipping", document_ids=[a], no_cache=True).json()
    assert {c["document_id"] for c in only_a["retrieved"]} <= {a}


# ---------- isolation ----------
def test_users_cannot_see_each_others_data(client, alice, bob):
    doc_id = upload(client, alice, "refunds.txt", REFUND).json()["id"]

    assert client.get(f"/documents/{doc_id}", headers=bob).status_code == 404
    assert client.delete(f"/documents/{doc_id}", headers=bob).status_code == 404
    assert client.get("/documents", headers=bob).json() == []
    assert ask(client, bob, "refund", document_ids=[doc_id]).status_code == 404

    body = ask(client, bob, "How many days do customers have to request a refund?").json()
    assert body["refused"] and body["retrieved"] == []  # Bob's search never touches Alice's chunks
    assert ask(client, alice, "How many days do customers have to request a refund?").json()["retrieved"]

    alice_conv = ask(client, alice, "refund policy").json()["conversation_id"]
    assert client.get(f"/conversations/{alice_conv}", headers=bob).status_code == 404
    assert ask(client, bob, "refund", conversation_id=alice_conv).status_code == 404


def test_same_file_can_be_uploaded_by_two_users(client, alice, bob):
    assert upload(client, alice, "refunds.txt", REFUND).status_code == 202
    assert upload(client, bob, "refunds.txt", REFUND).status_code == 202


# ---------- caching, deletion ----------
def test_cache_hit_and_invalidation(client, alice):
    upload(client, alice, "refunds.txt", REFUND)
    q = "How many days do customers have to request a refund?"
    assert ask(client, alice, q).json()["cached"] is False
    assert ask(client, alice, "  " + q.upper()).json()["cached"] is True  # normalised key
    upload(client, alice, "shipping.txt", SHIPPING)  # corpus changed
    assert ask(client, alice, q).json()["cached"] is False
    assert ask(client, alice, q, no_cache=True).json()["cached"] is False


def test_delete_removes_chunks_and_stale_answers(client, alice):
    doc_id = upload(client, alice, "refunds.txt", REFUND).json()["id"]
    q = "How many days do customers have to request a refund?"
    assert not ask(client, alice, q).json()["refused"]
    assert client.delete(f"/documents/{doc_id}", headers=alice).status_code == 204
    after = ask(client, alice, q).json()
    assert after["refused"] and after["cached"] is False  # not served from the stale cache


# ---------- limits, streaming, history ----------
def test_ask_rate_limit(client, alice, monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_ask_per_min", 2)
    assert ask(client, alice, "first question").status_code == 200
    assert ask(client, alice, "second question").status_code == 200
    third = ask(client, alice, "third question")
    assert third.status_code == 429 and "retry-after" in third.headers


def test_streaming_sends_tokens_then_done(client, alice):
    upload(client, alice, "refunds.txt", REFUND)
    r = ask(client, alice, "How many days do customers have to request a refund?", stream=True)
    assert r.headers["content-type"].startswith("text/event-stream")
    events = [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")]
    assert events[-1]["type"] == "done" and events[-1]["citations"]
    assert sum(e["type"] == "token" for e in events) >= 1


def test_conversation_history_and_usage(client, alice):
    upload(client, alice, "refunds.txt", REFUND)
    first = ask(client, alice, "How many days do customers have to request a refund?").json()
    ask(client, alice, "And how are refunds paid out?", conversation_id=first["conversation_id"])
    conv = client.get(f"/conversations/{first['conversation_id']}", headers=alice).json()
    assert [m["role"] for m in conv["messages"]] == ["user", "assistant", "user", "assistant"]
    assert conv["messages"][1]["citations"]

    usage = client.get("/usage", headers=alice).json()
    assert usage["queries"] == 2 and usage["avg_latency_ms"] >= 0
    assert client.delete(f"/conversations/{first['conversation_id']}", headers=alice).status_code == 204


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/health/ready").status_code == 200


def test_docs_example_has_no_fake_ids_and_404s_explain_themselves(client, alice):
    example = client.get("/openapi.json").json()["components"]["schemas"]["AskRequest"]["example"]
    assert list(example) == ["question"]  # so Swagger "Try it out" doesn't send placeholder UUIDs
    fake = "3fa85f64-5717-4562-b3fc-2b963f66afa6"
    r = ask(client, alice, "anything at all", conversation_id=fake)
    assert r.status_code == 404 and "Remove conversation_id" in r.json()["detail"]
    r = ask(client, alice, "anything at all", document_ids=[fake])
    assert r.status_code == 404 and "document_ids" in r.json()["detail"]


def test_swagger_example_is_minimal_and_errors_are_actionable(client, alice):
    from app.main import app

    example = app.openapi()["components"]["schemas"]["AskRequest"]["example"]
    assert set(example) == {"question"}  # no placeholder ids that would trigger 404s

    fake = "3fa85f64-5717-4562-b3fc-2c963f66afa6"  # the placeholder Swagger used to prefill
    bad_conv = ask(client, alice, "What skills are listed?", conversation_id=fake)
    assert bad_conv.status_code == 404 and "Remove conversation_id" in bad_conv.json()["detail"]
    bad_doc = ask(client, alice, "What skills are listed?", document_ids=[fake])
    assert bad_doc.status_code == 404 and "Remove document_ids" in bad_doc.json()["detail"]
