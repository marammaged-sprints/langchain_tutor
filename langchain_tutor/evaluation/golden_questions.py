"""Small, stable behavioral benchmark for the Think Python tutor."""

from __future__ import annotations


IN_SCOPE: list[tuple[str, tuple[str, ...]]] = [
    (
        "What is a program according to the book?",
        ("program", "instructions"),
    ),
    (
        "What is an assignment statement, and what does a variable refer to?",
        ("variable", "value"),
    ),
    (
        "What is the difference between a parameter and an argument?",
        ("parameter", "argument"),
    ),
    (
        "How does recursion work, and what is a base case?",
        ("recursion", "base case"),
    ),
    (
        "What is the difference between a list and a tuple?",
        ("tuple", "mutable", "immutable"),
    ),
    (
        "How do I catch an exception in Python?",
        ("try", "except"),
    ),
    (
        "What is a dictionary?",
        ("dictionary", "key"),
    ),
    (
        "What is aliasing in the context of lists?",
        ("alias", "same object", "reference"),
    ),
    (
        "How does a while statement work?",
        ("while", "condition", "loop"),
    ),
    (
        "What are list comprehensions and generator expressions?",
        ("list comprehension", "generator"),
    ),
]


OUT_OF_SCOPE: list[str] = [
    "What is the weather in Cairo today?",
    "How do I train a neural network with TensorFlow?",
    "Who wrote Pride and Prejudice?",
    "What is the current Gemini API pricing?",
    "Give me a SQL query that joins customers, orders, and products.",
]
