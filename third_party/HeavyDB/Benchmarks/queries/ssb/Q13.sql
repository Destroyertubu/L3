SELECT
  SUM(lo.lo_extendedprice * lo.lo_discount) AS revenue
FROM lineorder lo
WHERE lo.lo_orderdate >= 19940206
  AND lo.lo_orderdate <= 19940212
  AND lo.lo_discount >= 5
  AND lo.lo_discount <= 7
  AND lo.lo_quantity >= 26
  AND lo.lo_quantity <= 35;

