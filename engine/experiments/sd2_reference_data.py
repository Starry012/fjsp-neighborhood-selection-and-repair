"""Versioned SD2-mix generation and strict project JSON/FJS conversion."""

from hashlib import sha256
import json
from pathlib import Path


VERSION = "sd2-mix-v1"
SCALES = ((10, 5), (20, 5), (15, 10), (20, 10))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def generate(jobs, machines, split, index, attempt=0):
    import numpy as np
    if (type(jobs) is not int or jobs < 1 or type(machines) is not int or machines < 1
            or split not in ("train", "validation", "test", "calibration")
            or type(index) is not int or index < 0 or type(attempt) is not int or attempt < 0):
        raise ValueError("invalid SD2 generation specification")
    material = f"fjsp-daniel-ref-v1|{split}|{jobs}|{machines}|{index}|{attempt}"
    digest = sha256(material.encode()).hexdigest()
    rng = np.random.Generator(np.random.PCG64DXSM(int(digest, 16)))
    chains, options = [], []
    for job in range(jobs):
        operations = list(range(job * machines, (job + 1) * machines))
        chains.append({"job": job, "operations": operations})
        for op in operations:
            count = int(rng.integers(1, machines + 1))
            selected = sorted(int(m) for m in rng.choice(machines, size=count, replace=False))
            times = rng.integers(1, 100, size=count)
            options.extend({"operation": op, "machine": machine, "duration": int(duration)}
                           for machine, duration in zip(selected, times))
    return {"instance": f"sd2mix-{split}-{jobs}x{machines}-{index:04d}",
            "jobs": chains, "operations": options,
            "metadata": {"generator": VERSION, "machines": machines, "split": split,
                "index": index, "attempt": attempt, "seed_material": material,
                "seed_sha256": digest, "rng": "PCG64DXSM", "numpy": np.__version__}}


def signatures(data):
    from fjsp import parse_instance
    instance = parse_instance(data)
    machines = data["metadata"]["machines"]
    chains = [[sorted(instance.options[op].items()) for op in chain]
              for chain in instance.jobs.values()]
    if any(machine >= machines for row in instance.options.values() for machine in row):
        raise ValueError("machine outside declared count")
    return {"problem_sha256": sha256(canonical([machines, chains])).hexdigest(),
            "job_permutation_signature": sha256(canonical([machines, sorted(chains)])).hexdigest()}


def to_fjs(data):
    from fjsp import parse_instance
    instance = parse_instance(data)
    signatures(data)
    average = sum(map(len, instance.options.values())) / len(instance.options)
    lines = [f"{len(instance.jobs)} {data['metadata']['machines']} {average:.12g}"]
    for chain in instance.jobs.values():
        fields = [str(len(chain))]
        for op in chain:
            fields.append(str(len(instance.options[op])))
            for machine, duration in sorted(instance.options[op].items()):
                fields.extend((str(machine + 1), str(duration)))
        lines.append(" ".join(fields))
    return "\n".join(lines) + "\n"


def from_fjs(text, name):
    lines = text.splitlines()
    if not lines or len(lines[0].split()) not in (2, 3):
        raise ValueError("invalid FJS header")
    jobs, machines = map(int, lines[0].split()[:2])
    if jobs < 1 or machines < 1 or len(lines) != jobs + 1:
        raise ValueError("invalid FJS dimensions")
    chains, options, operation = [], [], 0
    for job, line in enumerate(lines[1:]):
        values = list(map(int, line.split()))
        if not values or values[0] < 1:
            raise ValueError("empty FJS job")
        position, chain = 1, []
        for _ in range(values[0]):
            if position >= len(values) or not 1 <= values[position] <= machines:
                raise ValueError("invalid option count")
            count = values[position]
            position += 1
            if position + 2 * count > len(values):
                raise ValueError("truncated FJS options")
            for _ in range(count):
                machine, duration = values[position:position + 2]
                if not 1 <= machine <= machines or duration < 1:
                    raise ValueError("invalid FJS machine/time")
                options.append({"operation": operation, "machine": machine - 1, "duration": duration})
                position += 2
            chain.append(operation)
            operation += 1
        if position != len(values):
            raise ValueError("trailing FJS fields")
        chains.append({"job": job, "operations": chain})
    data = {"instance": name, "jobs": chains, "operations": options, "metadata": {"machines": machines}}
    signatures(data)
    return data


def write_calibration(output, per_scale=10):
    """Only engineering data: no training, validation, test or public inputs."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows, seen, job_seen = [], set(), set()
    for jobs, machines in SCALES:
        for index in range(per_scale):
            attempt = 0
            while True:
                data = generate(jobs, machines, "calibration", index, attempt)
                stamps = signatures(data)
                if stamps["problem_sha256"] not in seen and stamps["job_permutation_signature"] not in job_seen:
                    break
                attempt += 1
                if attempt > 100:
                    raise ValueError("repeated generation collision")
            seen.add(stamps["problem_sha256"])
            job_seen.add(stamps["job_permutation_signature"])
            fjs = to_fjs(data)
            if signatures(from_fjs(fjs, data["instance"])) != stamps:
                raise ValueError("FJS roundtrip changed problem")
            name = data["instance"]
            raw = canonical(data) + b"\n"
            (output / f"{name}.json").write_bytes(raw)
            (output / f"{name}.fjs").write_text(fjs, encoding="utf-8", newline="\n")
            rows.append({"instance": name, "J": jobs, "M": machines, "N": jobs * machines,
                         "input": f"{name}.json", "fjs": f"{name}.fjs", **stamps,
                         "file_sha256": sha256(raw).hexdigest(), "seed_material": data["metadata"]["seed_material"],
                         "seed_sha256": data["metadata"]["seed_sha256"], "attempt": attempt})
    (output / "manifest.json").write_bytes(canonical({"version": VERSION, "split": "calibration", "cases": rows}) + b"\n")
    return rows
