def test_relationships_are_namespace_scoped_and_traverse_bounded_graph(db):
    db.repository.ensure_namespace("n1")
    db.repository.ensure_namespace("n2")
    a = db.repository.upsert_entity("n1", "termyte", "TermyteDB", "project")
    b = db.repository.upsert_entity("n1", "sqlite", "SQLite", "technology")
    c = db.repository.upsert_entity("n1", "fastembed", "FastEmbed", "technology")
    db.repository.add_relationship("n1", a, "uses", b)
    db.repository.add_relationship("n1", b, "supports", c)
    related = db.repository.related_entities("n1", a, depth=2)
    assert [row["predicate"] for row in related] == ["uses", "supports"]
    assert db.repository.related_entities("n2", a) == []
