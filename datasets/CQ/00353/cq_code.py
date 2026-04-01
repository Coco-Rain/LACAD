import cadquery as cq

globe = cq.Workplane("XY").sphere(10.125)
teh_box = cq.Workplane("XZ").box(13.98, 21.26, 21.26, centered=True)
result = globe.intersect(teh_box).edges().fillet(2.44)
cq.exporters.export(result, 'GT.stl')