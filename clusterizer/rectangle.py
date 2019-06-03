from functools import total_ordering
import functools
import numpy as np


@total_ordering
class Rectangle:
    def __init__(self, location_range=None, time_range=None, found_by=None):
        """
        :param location_range: (Optional) 2-Tuple containing left and right rectangle bounds
        :type location_range: tuple (float)

        :param time_range: (Optional) Tuple containing start and end dates of rectangle
        :type time_range: tuple (numpy.datetime64)
        """
        self.location_range = location_range
        self.time_range = time_range
        if found_by is None:
            self.found_by = set()
        else:
            self.found_by = set(found_by)

    @staticmethod
    def from_circuit_warning(circuit, warning_index, rectangle_width=None):
        """
        Create a new Rectangle instance that corresponds to one of the (DNV-GL) warnings of the MergedCircuit. `warning_index` chooses which warning to convert from. (A MergedCircuit has multiple warnings, in general.)

        :param circuit: Circuit object containing warning series to convert from
        :type circuit: class:`clusterizer.circuit.MergedCircuit`

        :param warning_index: The index of the warning to convert from
        :type warning_index: int

        :param rectangle_width: Width (m) of the Rectangle to create. When set to `None`, 1% of the circuit length is used (0.5% at both sides).
        :type rectangle_width: float, optional
        """
        w = circuit.warning.loc[warning_index]
        loc = w["Location in meters (m)"]
        dates = (w["Start Date/time (UTC)"], w["End Date/time (UTC)"])
        level = str(w["SCG warning level (1 to 3 or Noise)"])
        if rectangle_width is None:
            rectangle_width = circuit.circuitlength * 0.01

        loc_range = (loc - rectangle_width * .5, loc + rectangle_width * .5)

        return Rectangle(location_range=loc_range, time_range=dates, found_by={"DNV GL warning {}".format(level)})

    def get_width(self):
        """The distance in m between the two rectangle edges. `numpy.inf` if undefined.

        :rtype: float
        """
        if self.location_range is None:
            return np.inf
        return max(self.location_range) - min(self.location_range)

    def get_duration(self):
        """The time delta between the two rectangle edges. `numpy.inf` if undefined.

        :rtype: numpy.timedelta64
        """
        if self.time_range is None:
            return float("inf")
        return max(self.time_range) - max(self.time_range)

    # Wordt opgeroepen als je `str(een_rectangle)` of bv. `print(een_rectangle)` schrijft.
    def __str__(self):
        sentences = []
        if self.location_range is not None:
            sentences.append("{0:.0f}m to {1:.0f}m".format(*self.location_range))
        if self.time_range is not None:
            sentences.append("{0} until {1}".format(*self.time_range))
        if self.found_by:
            sentences.append("Found by: " + ', '.join(['%s']*len(self.found_by)) % tuple(self.found_by))
        return "; ".join(sentences)

    def __repr__(self):
        return str(self)

    def __lt__(self, other):
        """
        Provides the < comparison needed for a total ordering.
        Ordering is based on a 2-dimensional location, time grid.
        Time is seen as the more important factor in this comparison.
        If times can't be compared or are the same, location is used instead.
        This results in an ordering comparable to:
        (0, 0) < (0, 1) < (0, 2) < (1, 0) < (1, 1) < (2, 0)

        :param other: The other Rectangle to compare self with
        :type other: class:`clusterizer.rectangle.Rectangle`
        
        :return: True if self < other in specified total ordering, otherise False
        :rtype: bool
        """
        if self.time_range is None and other.time_range is None:
            if other.location_range is None:
                return False
            if self.location_range is None:
                return True
            return self.location_range < other.location_range
        if other.time_range is None:
            return Rectangle(self.location_range) < other
        if self.time_range is None:
            return self < Rectangle(other.location_range)
        if self.time_range == other.time_range:
            return Rectangle(self.location_range) < Rectangle(other.location_range)
        return self.time_range < other.time_range

    def __eq__(self, other):
        """
        Provides the == comparison needed for a total ordering.
        True if location_range and time_range for self and other are the same.
        
        :param other: The other Rectangle to compare self with
        :type other: class:`clusterizer.rectangle.Rectangle`
        
        :return: True if location and time ranges in self and other are the same, otherwise False
        :rtype: bool
        """
        return self.location_range == other.location_range and self.time_range == other.time_range

    def __hash__(self):
        """
        A hash function that computes a number based on the location and time ranges of the Rectangle
        Use case: Adding Rectangles to dictionaries, sets, etc.
        This function is cryptographically weak and should not be used for security purposes.

        :return: A hash of the Rectangle
        :rtype: int
        """
        return hash(self.location_range) + hash(self.time_range)

    def __and__(self, other):
        """
        Calculate the overlap between this rectangle and another rectangle.
        In doing so, it keeps track of the algorithms that found the Rectangles.
        If the rectangles are disjunct, return None.

        :param other: The other Rectangle to calculate the overlap with
        :type other: class:`clusterizer.rectangle.Rectangle` or None
        
        :return: The overlap between self and other
        :rtype: class:`clusterizer.rectangle.Rectangle` or None
        """
        def overlap(first, second):
            if first is not None and second is not None:
                left_bound = max(first[0], second[0])
                right_bound = min(first[1], second[1])
                if left_bound > right_bound:
                    return None
                return (left_bound, right_bound)
            if first is not None:
                return first
            return second

        if other is None:
            return None
        overlap_rectangle = Rectangle(overlap(self.location_range, other.location_range), overlap(self.time_range, other.time_range), self.found_by | other.found_by)
        if overlap_rectangle.location_range is None or overlap_rectangle.time_range is None:
            return None
        return overlap_rectangle

    def __rand__(self, other):
        """
        Right and. Needed to make things like 'None & Rectangle' work.
        """
        return self.__and__(other)

    def __mul__(self, other):
        """
        Calculate the overlap between this rectangle and another rectangle.
        Alternate alias for the & operator. Behaves exactly the same as &.

        :param other: The other Rectangle to calculate the overlap with
        :type other: class:`clusterizer.rectangle.Rectangle` or None
        
        :return: The overlap between self and other
        :rtype: class:`clusterizer.rectangle.Rectangle` or None
        """
        return self & other

    def overlap(self, other):
        """
        Alias for self & other.
        """
        return self & other

    def __or__(self, other):
        """
        Calculate the smallest rectangle which has both self and other as a subrectangle.
        This is also known as the bounding box containing self and other.
        If other is None, return None.

        :param other: The other Rectangle to calculate to bounding box with
        :type other: class:`clusterizer.rectangle.Rectangle` or None

        :return: The bounding box of self and other
        :rtype: class:`clusterizer.rectangle.Rectangle` or None
        """
        def least_common_superrange(first, second):
            if first is not None and second is not None:
                left_bound = min(first[0], second[0])
                right_bound = max(first[1], second[1])
                return (left_bound, right_bound)
            return None
        if other is None:
            return None
        return Rectangle(least_common_superrange(self.location_range, other.location_range), least_common_superrange(self.time_range, other.time_range), self.found_by | other.found_by)

    def __ror__(self, other):
        """
        Right or. Needed to make things like 'None | Rectangle' work.
        """
        return self.__or__(other)

    def superrectangle(self, other):
        """
        Alias for self | other.
        """
        return self | other

    def bounding_box(self, other):
        """
        Alias for self | other.
        """
        return self | other

    def disjunct(self, other):
        """
        Determines if two Rectangles are disjunct. That is, disjunct(self, other) returns:
        - True if self and other do not overlap
        - False if self and other overlap
        So: disjunct(self, other) is another way (more efficient) way to write `(self & other) is None`

        :param other: The other Rectangle to compare self to
        :type other: class:`clusterizer.rectangle.Rectangle` or None
        
        :return: True if self and other are disjunct, False otherwise
        :rtype: bool
        """
        def disjunct_range(first, second):
            if first is not None and second is not None:
                return first[1] <= second[0] or second[1] <= first[0]
            return False
        if other is None:
            return True
        return disjunct_range(self.location_range, other.location_range) or disjunct_range(self.time_range, other.time_range)

    def __sub__(self, other):
        """
        Calculate self without other, overloading the - operator.
        This is similar to the set-theoretic operation: self \ other.
        Unlike &, | and +, - is not commutative. self - other is not equal to other - self.
        Returns a set containing Rectangle objects. Together, these Rectangles represent self - other.
        
        :param other: The other Rectangle to substract from self
        :rtype other: class:`clusterizer.rectangle.Rectangle`

        :return: set containing rectangles 
        :rtype: set of class:`clusterizer.rectangle.Rectangle`
        """
        def empty_range(r):
            if r is None:
                return False
            return r[0] == r[1]

        def empty_rectangle(rectangle):
            if rectangle is None:
                return True
            if empty_range(rectangle.location_range) or empty_range(rectangle.time_range):
                return True
            return False

        if self.disjunct(other):
            return set([self])
        #self - other is the same as self - (self & other), since (other - self) and self are disjunct
        overlap = self & other
        #For the range [ ] and overlap range { }, we can represent all cases with the model:
        # [ { } ], where we need to calculate the 3 ranges [ {, { }, and } ]
        left_locations = (self.location_range[0], overlap.location_range[0])   # [ {
        middle_locations = overlap.location_range                              # { }
        right_locations = (overlap.location_range[1], self.location_range[1])  # } ]
        if self.time_range is not None and overlap.time_range is not None:
            lower_times = (self.time_range[0], overlap.time_range[0])          # [ {
            middle_times = overlap.time_range                                  # { }
            upper_times = (overlap.time_range[1], self.time_range[1])          # } ]
        else:
            lower_times, middle_times, upper_times = self.time_range, self.time_range, self.time_range
        #Split rectangle into 9 regions in a 3x3 grid, of which the center region ({ }, { }) is never needed
        #Depending on where self and other overlap, some of these can be empty
        lu = Rectangle(left_locations, upper_times, found_by=self.found_by)     # [ {, [ {
        mu = Rectangle(middle_locations, upper_times, found_by=self.found_by)   # { }, [ {
        ru = Rectangle(right_locations, upper_times, found_by=self.found_by)    # } ], [ {
        lm = Rectangle(left_locations, middle_times, found_by=self.found_by)    # [ {, { }
        rm = Rectangle(right_locations, middle_times, found_by=self.found_by)   # } ], { }
        ll = Rectangle(left_locations, lower_times, found_by=self.found_by)     # [ {, } ]
        ml = Rectangle(middle_locations, lower_times, found_by=self.found_by)   # { }, } ]
        rl = Rectangle(right_locations, lower_times, found_by=self.found_by)    # } ], } ]
        mini_rectangles = [lu, mu, ru, lm, rm, ll, ml, rl]
        result = set()
        #Filter out the empty regions of the 3x3 grid        
        for c in mini_rectangles:
            if not empty_rectangle(c):
                result.add(c)
        return result

    def __add__(self, other):
        """
        Calculated the rich set-theoretic union of two Rectangles, overloading the + operator.
        In doing so, it keeps track of which algorithms found the rectangles.
        The best analogy for the result is a Venn diagram. The overlapping parts are where the algorithms agree. Unlike &, + also remembers where the algorithms disagree.
        Returns a set containing Rectangle objects. Together, these Rectangles represent self + other.
        
        :param other: The other Rectangle to calculate the union with
        :type other: class:`clusterizer.rectangle.Rectangle`
        
        :return: The set-theoretic union of self and other
        :rtype: set of class:`clusterizer.rectangle.Rectangle`
        """
        if other is None:
            return set([self])
        if self.disjunct(other):
            return set([self, other])
        overlap = self & other
        smo = self - other
        oms = other - self
        return set([overlap]) | smo | oms



    def get_partial_discharges(self, circuit):
        """Returns all PDs that lie in the Rectangle."""
        return circuit.pd[circuit.pd_occured].loc[self.get_partial_discharge_mask(circuit)]

    def get_partial_discharge_mask(self, circuit):
        """Returns a boolean array which indicates, for each PD, whether it lies in the Rectangle."""
        locations = circuit.pd["Location in meters (m)"][circuit.pd_occured]
        times = circuit.pd["Date/time (UTC)"][circuit.pd_occured]

        return functools.reduce(np.logical_and, [
            self.location_range[0] < locations,
            self.location_range[1] >= locations,
            self.time_range[0] < times,
            self.time_range[1] >= times])
