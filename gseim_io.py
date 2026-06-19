from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


@dataclass
class OutputBlock:
    dat_filename: str           
    variables: list[str] = field(default_factory=list)  
    dat_path: Path | None = None  


class GseimInFileError(ValueError):
    """Raised when a .in file can't be parsed the way we expect."""


def parse_in_file(in_file_path) -> tuple[int, list[OutputBlock]]:
    in_file_path = Path(in_file_path).expanduser()
    if not in_file_path.exists():
        raise GseimInFileError(f".in file not found: {in_file_path}")

    lines = in_file_path.read_text().splitlines()
    base_dir = in_file_path.parent

    n_solve_blocks = sum(1 for ln in lines if "begin_solve" in ln)

    output_blocks: list[OutputBlock] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if "begin_output" in line:
            block_start = i
            next_line = lines[i + 1] if i + 1 < n else ""
            if ".dat" in next_line:
                eq_pos = next_line.index("=")
                dat_end = next_line.index(".dat") + len(".dat")
                dat_filename = next_line[eq_pos + 1:dat_end].strip()

                j = i + 1
                variables: list[str] = []
                while j < n and "end_output" not in lines[j]:
                    if "variables:" in lines[j]:
                        after_colon = lines[j].split(":", 1)[1]
                        variables = after_colon.strip().split()
                    j += 1
                if j >= n:
                    raise GseimInFileError(
                        f"begin_output at line {block_start + 1} has no matching end_output"
                    )

                output_blocks.append(OutputBlock(
                    dat_filename=dat_filename,
                    variables=variables,
                    dat_path=(base_dir / dat_filename).resolve(),
                ))
                i = j
        i += 1

    return n_solve_blocks, output_blocks


def load_dat(dat_path) -> np.ndarray:
    dat_path = Path(dat_path).expanduser()
    if not dat_path.exists():
        raise FileNotFoundError(f".dat file not found: {dat_path}")

    data = np.loadtxt(dat_path)
    if data.size == 0:
        raise ValueError(f".dat file has no data: {dat_path}")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def column_labels(variables: list[str], n_col: int) -> list[str]:
    labels = list(variables)
    if len(labels) == n_col - 1:
        labels.insert(0, "time")
    if len(labels) != n_col:
        labels = [f"col{i}" for i in range(n_col)]
    return labels

def main():
    in_file = "Animation_boost_1.in"
    n_solve, output_blocks = parse_in_file(in_file)
    print(f"Number of solve blocks: {n_solve}")
    for block in output_blocks:
        print(f"Dat file: {block.dat_filename}")
        print(f"Variables: {block.variables}")
        print(f"Dat path: {block.dat_path}")
        data = load_dat(block.dat_path)
        print(f"Data shape: {data.shape}")
        labels = column_labels(block.variables, data.shape[1])
        print(f"Column labels: {labels}")


