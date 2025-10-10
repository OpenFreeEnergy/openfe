**Added:**

* <news item>

**Changed:**

* Units must be explicitly assigned when defining ``Settings`` parameters, and values will be converted to match the default units for a given field. For example, use ``1.0 * units.bar`` or ``"1 bar"`` for pressure, and ``300 * unit.kelvin`` or ``"300 kelvin"`` for temperature.
* For protocol developers: ``FloatQuantity`` is no longer supported. Instead, use `GufeQuantity` and `specify_quantity_units()` to make a `TypeAlias`.


**Deprecated:**

* <news item>

**Removed:**

* <news item>

**Fixed:**

* <news item>

**Security:**

* <news item>
