SELECT
  SUM(lo.lo_extendedprice * lo.lo_discount) AS revenue
FROM lineorder lo
WHERE lo.lo_orderdate >= 19940101
  AND lo.lo_orderdate <= 19940131
  AND lo.lo_discount >= 4
  AND lo.lo_discount <= 6
  AND lo.lo_quantity >= 26
  AND lo.lo_quantity <= 35;

