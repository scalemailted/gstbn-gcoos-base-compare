import math 
import haversine as hs


class Location:
    #Constructor - initializes an instance of a location
    def __init__(self, x, y, z=None, precision=1):
        self.x = round(x,precision)
        self.y = round(y,precision)
        self.z = round(z,precision) if z else None 
        self.ndim = 2 if (z == None) else 3
    
    #toString - returns location state in json-formatted string
    def __str__(self):
        z_str = "" if (self.z == None) else f', "z": {self.z}'
        return f'{{"x": {self.x}, "y": {self.y}{z_str}}}'
    
    #equals - returns is this location the same as another
    def __eq__(self, other):
        is_equal = isinstance(other, Location)
        is_equal = is_equal and (self.x == other.x)
        is_equal = is_equal and (self.y == other.y) 
        is_equal = is_equal and (self.z == other.z)
        return is_equal
    
    def __hash__(self):
        return hash(str(self))
    
    #distance - standard formula based in cartessian space  
    def distance(self, other):
        if isinstance(other, Location):
            dx = (self.x - other.x)**2
            dy = (self.y - other.y)**2
            dz = 0 if (self.z == None) else (self.z - other.z)**2
            d = math.sqrt(dx + dy + dz)
            return d
    
    #in_range - checks if this location is within range of another location
    def in_range(self, other, range):
        if isinstance(other, Location):
            return self.distance(other) <= range
    
    #tuple - returns this location as a tuple value
    def tuple(self):
        return (self.x, self.y)
    
    #move_by - adds (dx, dy) to the current (x, y) values
    def move_by(self,dx=0,dy=0):
        self.x += dx
        self.y += dy
    
    #clone - returns a new copy of this location object
    def clone(self):
        return Location(self.x, self.y, self.z)



#Geolocation - subclass of location, overrides the distance formula for lons/lats 
class Geolocation(Location):
    def distance(self, other):
        if isinstance(other, Geolocation):
            loc1 = (self.x, self.y)
            loc2 = (other.x, other.y)
            return hs.haversine(loc1 ,loc2)