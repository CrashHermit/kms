"""Application services for user and learning workflows.

Services receive repositories assembled by ``kms2.composition`` and coordinate
domain transitions. Repositories own database sessions and queries; services
do not construct dependencies or own database/model runtime lifetimes.
"""
