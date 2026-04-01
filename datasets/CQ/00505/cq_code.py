import cadquery as cq

sphere1 = cq.Workplane("XY").sphere(5)
sphere2 = cq.Workplane("XY").transformed(offset=(7, 0, 0)).sphere(5)
result = sphere1.union(sphere2, clean=True)
cq.exporters.export(result, 'GT.stl')