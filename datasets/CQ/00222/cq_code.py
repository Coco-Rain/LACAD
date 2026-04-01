import cadquery as cq

result = (
cq.Workplane("XY")
.workplane(offset=0)
.center(0, 0)
.cylinder(height=5, radius=2)
)
cq.exporters.export(result, 'GT.stl')