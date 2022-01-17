from pathlib import Path

import click


@click.group()
def cli():
    pass


@cli.command()
@click.argument("name")
def generate_docs(name):
    from ssa_oho_utils.process_measures.interface.documentation import (
        render_doc_for_analytic,
    )
    from ssa_oho_utils.process_measures.interface import (
        ProcessMeasureRunnerInterface,
    )
    from ssa_oho_utils.process_measures import (  # noqa: F401
        get_cases_at_least_k_transfers,
        get_claims_moved_by_time_interval,
        get_count_ngrams,
        get_deviations_from_benchmarks_by_claim,
        get_deviations_from_benchmarks_by_code,
        get_fraction_of_codes_exceeding_benchmarks_by_threshold,
        get_frequency_of_referrals_deferrals,
        get_loop_counts,
        get_median_time_by_case_age_status,
        get_office_ref_mapping_table,
        get_office_types_used_by_case,
        get_page_count_statistics,
        get_percent_cases_by_office,
        get_percent_non_disability_claims,
        get_prop_claims_returning_from_temp_transfer_above_below_equiv_time,
        get_proportion_claims_exceed_over1_benchmark,
        get_proportion_claims_follow_fifo,
        get_proportion_milestones_exceeded,
        get_status_code_stats,
        get_time_spent_in_status_after_given_days,
    )
    from ssa_oho_utils.process_measures.interface import _examples  # noqa: F401

    if name not in ProcessMeasureRunnerInterface.all_analytics:
        raise KeyError(
            f"{name} not found in {list(ProcessMeasureRunnerInterface.all_analytics)}"
        )
    analytic = ProcessMeasureRunnerInterface.all_analytics[name]
    documentation = render_doc_for_analytic(analytic)
    output_path = (
        Path(__file__).parent.parent
        / f"docs/source/process_measures/{analytic.slug}.rst"
    )
    open(output_path, "w", encoding="utf8", newline="\n").write(documentation)
    click.echo(f"Documentation written to {output_path}")
    click.echo(
        "If this is the first time you've generated these docs, don't forget to add them to an index table of contents"
    )


if __name__ == "__main__":
    cli()
