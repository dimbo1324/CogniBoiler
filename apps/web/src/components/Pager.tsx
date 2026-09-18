export function Pager({
  offset,
  limit,
  total,
  onChange,
}: {
  offset: number;
  limit: number;
  total: number;
  onChange: (offset: number) => void;
}) {
  const first = total === 0 ? 0 : offset + 1;
  const last = Math.min(offset + limit, total);
  return (
    <div className="pager" aria-label="Pages">
      <button
        type="button"
        disabled={offset === 0}
        onClick={() => {
          onChange(Math.max(offset - limit, 0));
        }}
      >
        Previous
      </button>
      <span>
        {first}–{last} of {total}
      </span>
      <button
        type="button"
        disabled={offset + limit >= total}
        onClick={() => {
          onChange(offset + limit);
        }}
      >
        Next
      </button>
    </div>
  );
}
