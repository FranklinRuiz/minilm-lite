package io.github.franklinruiz.classifier;

/**
 * Utility class for validating input parameters.
 * Provides static methods to ensure the correctness of input values, such as checking for nulls,
 * ensuring a value is greater than zero, or verifying that a number lies within a specified range.
 */
public class ValidationUtils {

    private ValidationUtils() {
        throw new IllegalStateException("ValidationUtils class");
    }

    /**
     * Ensures that an integer value is greater than zero.
     *
     * @param i    The integer value to validate.
     * @param name The name of the parameter, used in the exception message if validation fails.
     * @return The validated integer value.
     * @throws IllegalArgumentException if the value is less than or equal to zero.
     */
    public static int ensureGreaterThanZero(int i, String name) {
        if (i <= 0) {
            throw new IllegalArgumentException(name + " must be greater than zero, but is: " + i);
        }
        return i;
    }

    /**
     * Ensures that a double value lies within a specified range [min, max].
     *
     * @param d    The double value to validate.
     * @param min  The minimum allowable value (inclusive).
     * @param max  The maximum allowable value (inclusive).
     * @param name The name of the parameter, used in the exception message if validation fails.
     * @return The validated double value.
     * @throws IllegalArgumentException if the value is outside the specified range.
     */
    public static double ensureBetween(double d, double min, double max, String name) {
        if (d < min || d > max) {
            throw new IllegalArgumentException(name + " must be between " + min + " and " + max + ", but is: " + d);
        }
        return d;
    }

    /**
     * Ensures that an object is not null.
     *
     * @param object The object to validate.
     * @param name   The name of the parameter, used in the exception message if validation fails.
     * @throws IllegalArgumentException if the object is null.
     */
    public static void ensureNotNull(Object object, String name) {
        if (object == null) {
            throw new IllegalArgumentException(name + " cannot be null");
        }
    }
}