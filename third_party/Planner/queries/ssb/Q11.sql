SELECT
  SUM(lo.lo_extendedprice * lo.lo_discount) AS revenue
FROM lineorder lo
WHERE lo.lo_orderdate >= 19930000
  AND lo.lo_orderdate <= 19940000
  AND lo.lo_discount >= 1
  AND lo.lo_discount <= 3
  AND lo.lo_quantity < 25;

