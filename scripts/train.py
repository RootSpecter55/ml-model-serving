"""Train a model from the command line (same code path as POST /training/runs).

    uv run python -m scripts.train                                  # logistic regression
    uv run python -m scripts.train --algorithm random_forest --param n_estimators=200
    uv run python -m scripts.train --promote                        # also promote to production

Reads settings from the environment / .env, like the API.
"""

import argparse
import asyncio
import json

from dotenv import load_dotenv


def parse_param(text: str) -> tuple[str, object]:
    key, _, value = text.partition("=")
    try:
        return key, json.loads(value)  # numbers and null become their JSON types
    except json.JSONDecodeError:
        return key, value


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default=None)
    parser.add_argument("--algorithm", default="logistic_regression")
    parser.add_argument("--param", action="append", default=[], help="key=value, repeatable")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--promote", action="store_true", help="promote the new version")
    args = parser.parse_args()

    load_dotenv()
    # Imported after .env is loaded: settings are read when the container is built.
    from app.container import build_container
    from app.core.config import get_settings
    from app.modules.training.schemas import TrainingRequest

    container = build_container(get_settings())
    request = TrainingRequest(
        name=args.name,
        algorithm=args.algorithm,
        params=dict(parse_param(p) for p in args.param),
        test_size=args.test_size,
        random_state=args.random_state,
    )
    result = await container.training_service.train(request)
    version = result.model_version
    print(f"Registered {version.name} v{version.version} ({version.id})")
    print(json.dumps(version.metrics, indent=2))
    if args.promote:
        await container.registry_service.promote(version.id, "production")
        print("Promoted to production")


if __name__ == "__main__":
    asyncio.run(main())
