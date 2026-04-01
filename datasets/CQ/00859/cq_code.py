import cadquery as cq

horisontal_tube = cq.Workplane("XZ").cylinder(70, 11)
vertical_tube = (
cq.Workplane("XY")
.cylinder(50, 11)
.translate((0, 0, 25))
)
result = horisontal_tube.union(vertical_tube)
cq.exporters.export(result, 'GT.stl')