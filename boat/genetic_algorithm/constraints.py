"""Classes for handling mutation constraints in genetic algorithms."""

import random


class MutationConstraintHandler:
    """Base class for handling mutation constraints."""

    def adjust_vocabulary(self, sequence: str, aa_vocabulary: dict[int, str], **kwargs) -> dict[int, str]:
        """Return the (potentially adjusted) vocabulary for a given sequence.

        Args:
            sequence : str
                The sequence to be mutated.
            aa_vocabulary : dict[int, str]
                The original amino acid vocabulary.
        """
        return aa_vocabulary

    def check_constraint_met(self, sequence: str) -> bool:
        """Check if the sequence satisfies the constraint."""
        return True

    def repair_sequence(self, sequence: str) -> str:
        """Repair the sequence to satisfy the constraint."""
        return sequence


class MaxMutationConstraintHandler(MutationConstraintHandler):
    """Constraint handler for limiting total number of mutations."""

    def __init__(self, parental_sequence: str, max_mutations: int):
        """Initialize the max mutation constraint handler.

        Args:
            parental_sequence : str
                The parental sequence to be used as a reference for counting mutations.
            max_mutations : int
                The maximum number of mutations allowed.
        """
        self.parental_sequence = parental_sequence
        self.max_mutations = max_mutations

    def adjust_vocabulary(self, sequence: str, aa_vocabulary: dict[int, str], **kwargs) -> dict[int, str]:
        """Restrict vocabulary if max mutation limit is reached."""
        adjusted_vocab = aa_vocabulary.copy()

        # Count current mutations
        n_muts = sum(1 for i in range(len(sequence)) if sequence[i] != self.parental_sequence[i])

        # If at limit, only allow mutations at already-mutated positions
        if n_muts >= self.max_mutations:
            for i in range(len(sequence)):
                if sequence[i] == self.parental_sequence[i]:
                    # Remove this position from vocabulary
                    if i in adjusted_vocab:
                        del adjusted_vocab[i]

        return adjusted_vocab

    def check_constraint_met(self, sequence: str) -> bool:
        """Check if the sequence satisfies the max mutation constraint.

        Args:
            sequence : str
                The sequence to check.
        """
        n_muts = sum(1 for i in range(len(sequence)) if sequence[i] != self.parental_sequence[i])
        return n_muts <= self.max_mutations

    def repair_sequence(self, sequence: str) -> str:
        """Repair the sequence by reverting mutations if the max mutation limit is exceeded.

        Args:
            sequence : str
                The sequence to repair.

        Returns
        -------
            str
                The repaired sequence.
        """
        if self.check_constraint_met(sequence):
            return sequence  # No repair needed

        # Get revertable positions
        revertable_positions = self._get_revertable_positions(sequence)
        random.shuffle(revertable_positions)

        # Revert mutations until within limit
        sequence_list = list(sequence)
        n_muts = len(revertable_positions)
        while n_muts > self.max_mutations and revertable_positions:
            pos_to_revert = revertable_positions.pop()
            sequence_list[pos_to_revert] = self.parental_sequence[pos_to_revert]
            n_muts -= 1

        return "".join(sequence_list)

    def _get_revertable_positions(self, sequence: str) -> list[int]:
        """Get positions that can be reverted to parental amino acids in case the max mutation limit is exceeded.

        Args:
            sequence : str
                The sequence to check.

        Returns
        -------
            list of int
                List of positions that can be reverted.
        """
        revertable_positions = []
        for i in range(len(sequence)):
            if sequence[i] != self.parental_sequence[i]:
                revertable_positions.append(i)
        return revertable_positions


class CDRMutationConstraintHandler(MutationConstraintHandler):
    """Constraint handler for limiting mutations per CDR region.

    Note that if there are mutations defined outside of the CDR regions, these are maintained.
    """

    def __init__(
        self, parental_sequence: str, max_mutations_per_cdr: dict[str, int], cdr_positions: dict[str, list[int]]
    ):
        """Initialize the CDR mutation constraint handler.

        Args:
            parental_sequence : str
                The parental sequence to be used as a reference for counting mutations.
            max_mutations_per_cdr : dict[str, int]
                The maximum number of mutations allowed per CDR region.
            cdr_positions : dict[str, list[int]]
                A dictionary defining the positions of CDR regions in the sequence.
        """
        self.parental_sequence = parental_sequence
        self.max_mutations_per_cdr = max_mutations_per_cdr
        self.cdr_positions = cdr_positions

    def adjust_vocabulary(self, sequence: str, aa_vocabulary: dict[int, str], **kwargs) -> dict[int, str]:
        """Restrict vocabulary if CDR mutation limits are reached."""
        adjusted_vocab = aa_vocabulary.copy()

        for cdr_name, positions in self.cdr_positions.items():
            # Count current mutations in this CDR
            n_muts = sum(1 for pos in positions if sequence[pos] != self.parental_sequence[pos])

            # If at limit, only allow mutations at already-mutated positions
            if n_muts >= self.max_mutations_per_cdr[cdr_name]:
                for pos in positions:
                    if sequence[pos] == self.parental_sequence[pos]:
                        # Remove this position from vocabulary
                        if pos in adjusted_vocab:
                            del adjusted_vocab[pos]

        return adjusted_vocab

    def check_constraint_met(self, sequence: str) -> bool:
        """Check if the sequence satisfies the CDR mutation constraints.

        Args:
            sequence : str
                The sequence to check.
        """
        for cdr_name, positions in self.cdr_positions.items():
            n_muts = sum(1 for pos in positions if sequence[pos] != self.parental_sequence[pos])
            if n_muts > self.max_mutations_per_cdr[cdr_name]:
                return False
        return True

    def _get_revertable_positions(self, sequence: str) -> dict[str, list[int]]:
        """Get positions that can be reverted to parental amino acids in case the CDR mutation limits are exceeded.

        Args:
            sequence : str
                The sequence to check.

        Returns
        -------
            dict[str, list[int]]
                Dictionary of CDR names to lists of positions that can be reverted.
        """
        revertable_positions = {}
        for cdr_name, positions in self.cdr_positions.items():
            revertable_positions[cdr_name] = []
            for pos in positions:
                if sequence[pos] != self.parental_sequence[pos]:
                    revertable_positions[cdr_name].append(pos)

            # shuffle positions to ensure random selection during repair
            random.shuffle(revertable_positions[cdr_name])

            if len(revertable_positions[cdr_name]) <= self.max_mutations_per_cdr[cdr_name]:
                # No need to revert any positions when within limit
                revertable_positions[cdr_name] = []
        return revertable_positions

    def repair_sequence(self, sequence: str) -> str:
        """Repair the sequence by reverting mutations if the CDR mutation limits are exceeded.

        Args:
            sequence : str
                The sequence to repair.

        Returns
        -------
            str
                The repaired sequence.
        """
        if self.check_constraint_met(sequence):
            return sequence  # No repair needed

        # Get revertable positions
        revertable_positions = self._get_revertable_positions(sequence)
        sequence_list = list(sequence)
        for cdr_name, positions in revertable_positions.items():
            n_muts = len(positions)
            while n_muts > self.max_mutations_per_cdr[cdr_name] and positions:
                pos_to_revert = positions.pop()
                sequence_list[pos_to_revert] = self.parental_sequence[pos_to_revert]
                n_muts -= 1

        return "".join(sequence_list)


class CompositeConstraintHandler(MutationConstraintHandler):
    """Handler that applies multiple constraints in sequence."""

    def __init__(self, handlers: list[MutationConstraintHandler]):
        """Initialize with a list of constraint handlers.

        Args:
            handlers : list[MutationConstraintHandler]
                List of constraint handlers to apply in order.
        """
        self.handlers = handlers

    def adjust_vocabulary(self, sequence: str, aa_vocabulary: dict[int | str, str], **kwargs) -> dict[int | str, str]:
        """Apply all constraint handlers sequentially."""
        adjusted_vocab = aa_vocabulary.copy()
        for handler in self.handlers:
            adjusted_vocab = handler.adjust_vocabulary(sequence, adjusted_vocab, **kwargs)
        return adjusted_vocab

    def check_constraint_met(self, sequence: str) -> bool:
        """Check if the sequence satisfies all constraints.

        Args:
            sequence : str
                The sequence to check.
        """
        return all(handler.check_constraint_met(sequence) for handler in self.handlers)

    def repair_sequence(self, sequence: str) -> str:
        """Repair the sequence by applying all constraint handlers sequentially.

        Args:
            sequence : str
                The sequence to repair.

        Returns
        -------
            str
                The repaired sequence.
        """
        if self.check_constraint_met(sequence):
            return sequence  # No repair needed

        priority_order = (CDRMutationConstraintHandler, MaxMutationConstraintHandler)

        if any(not isinstance(h, priority_order) for h in self.handlers):
            raise ValueError(
                "All handlers must be of type MaxMutationConstraintHandler or CDRMutationConstraintHandler."
            )

        # Sort handlers by priority. CDR constaints are handled first, as they are more specific.
        self.handlers.sort(
            key=lambda h: priority_order.index(type(h)) if type(h) in priority_order else len(priority_order)
        )

        repaired_sequence = sequence
        for handler in self.handlers:
            if not handler.check_constraint_met(repaired_sequence):
                repaired_sequence = handler.repair_sequence(repaired_sequence)
        return repaired_sequence


def create_constraint_handler(
    parental_sequence: str | None = None,
    max_mutations: int | None = None,
    max_mutations_per_cdr: dict[str, int] = {},
    cdr_positions: dict[str, list[int]] = {},
) -> MutationConstraintHandler:
    """Create a composite constraint handler based on provided constraints.

    Args:
        parental_sequence : str | None
            The parental sequence to be used as a reference for counting mutations.
        max_mutations : int | None
            The maximum number of mutations allowed.
        max_mutations_per_cdr : dict[str, int] | None
            The maximum number of mutations allowed per CDR region.
        cdr_positions : dict[str, list[int]] | None
            A dictionary defining the positions of CDR regions in the sequence.

    Returns
    -------
        MutationConstraintHandler
            The composite constraint handler.
    """
    handlers = []
    if parental_sequence is not None and max_mutations is not None:
        handlers.append(MaxMutationConstraintHandler(parental_sequence, max_mutations))
    if parental_sequence is not None and max_mutations_per_cdr and cdr_positions:
        handlers.append(CDRMutationConstraintHandler(parental_sequence, max_mutations_per_cdr, cdr_positions))
    if not handlers:
        return MutationConstraintHandler()
    if len(handlers) == 1:
        return handlers[0]
    return CompositeConstraintHandler(handlers)
