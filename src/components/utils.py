from collections import Counter

from src.interfaces import Stop


def validate_route_stops(
    route_stops: list[Stop],
    *,
    base_drop_ids: set[str],  # request_ids of onboard passengers (drop-offs only)
    new_pair_ids: set[str],  # request_ids of new requests inserted as (pickup, drop)
    initial_occupancy: int,  # len(passengers_onboard)
    capacity: int,
    assert_unique_objects: bool = False,  # optional: ensure no identical Stop objects repeated
) -> None:
    """
    Raises ValueError with a clear message if invalid.
    Rules:
      - All base IDs appear exactly once as DROP stops; no pickups for base IDs.
      - Each new id appears exactly once as PICKUP and once as DROP.
      - For each new id: pickup index < drop index (precedence).
      - Capacity respected during a left-to-right scan:
            occ = initial_occupancy
            +1 at pickup(new id), -1 at any drop (base or new)
        (You can’t drop a rider before they’re onboard.)
    """
    # 0) Basic composition checks
    if assert_unique_objects:
        c = Counter(route_stops)
        dup = [s for s, k in c.items() if k > 1]
        if dup:
            raise ValueError(f"Route contains duplicated Stop objects: {dup[:3]}...")

    ids_in_route = {s.request_id for s in route_stops}
    allowed_ids = base_drop_ids | new_pair_ids
    extra = ids_in_route - allowed_ids
    missing_base = base_drop_ids - ids_in_route
    if extra:
        raise ValueError(f"Route contains unknown request IDs: {sorted(extra)}")
    if missing_base:
        raise ValueError(f"Route missing committed drop(s) for: {sorted(missing_base)}")

    # 1) Base drop-offs: must be exactly one DROP each, no PICKUP
    base_drop_count = Counter()
    for s in route_stops:
        if s.request_id in base_drop_ids:
            if s.is_pickup:
                raise ValueError(f"Found pickup for committed base id={s.request_id}")
            base_drop_count[s.request_id] += 1
    bad_mult = [rid for rid, k in base_drop_count.items() if k != 1]
    if bad_mult or len(base_drop_count) != len(base_drop_ids):
        raise ValueError("Each committed base id must appear exactly once as a drop.")

    # 2) New pairs: exactly one PICKUP and one DROP, and precedence
    seen_pu, seen_do = set(), set()
    first_idx, last_idx = {}, {}
    for i, s in enumerate(route_stops):
        if s.request_id in new_pair_ids:
            if s.is_pickup:
                if s.request_id in seen_pu:
                    raise ValueError(f"Duplicate pickup for {s.request_id}")
                seen_pu.add(s.request_id)
                first_idx.setdefault(s.request_id, i)
            else:
                if s.request_id in seen_do:
                    raise ValueError(f"Duplicate drop for {s.request_id}")
                seen_do.add(s.request_id)
                last_idx[s.request_id] = i

    missing_pu = new_pair_ids - seen_pu
    missing_do = new_pair_ids - seen_do
    if missing_pu or missing_do:
        raise ValueError(
            f"New pairs must have one pickup & one drop. "
            f"Missing pickup: {sorted(missing_pu)}, missing drop: {sorted(missing_do)}"
        )

    for rid in new_pair_ids:
        if first_idx[rid] >= last_idx[rid]:
            raise ValueError(f"Precedence violated for {rid}: drop precedes pickup")

    # 3) Capacity scan (and sanity: can’t drop someone not onboard)
    occ = initial_occupancy
    onboard_ids = (
        set()
    )  # track only new ids picked in this route (base riders already onboard)
    for s in route_stops:
        rid = s.request_id
        if s.is_pickup:
            # Only new requests have pickups in this model
            if rid not in new_pair_ids:
                raise ValueError(f"Unexpected pickup for non-new id={rid}")
            if rid in onboard_ids:
                raise ValueError(f"Duplicate pickup encountered for {rid}")
            onboard_ids.add(rid)
            occ += 1
        else:
            # drop for either base id or new id (if previously picked)
            if rid in new_pair_ids:
                if rid not in onboard_ids:
                    raise ValueError(f"Drop before pickup for {rid}")
                onboard_ids.remove(rid)
            # base drop is always valid here
            occ -= 1

        if occ > capacity:
            raise ValueError(f"Capacity exceeded: occ={occ} > {capacity}")
        if occ < 0:
            raise ValueError(f"Negative occupancy encountered at stop {s}")

    # Passed all checks
    return
