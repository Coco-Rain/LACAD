import cadquery as cq

result = (
cq.Workplane("XY")
.workplane(offset=0)
.center(2, 2)
.cylinder(height=10, radius=4, direct=(1,0,0))
)
cq.exporters.export(result, 'GT.stl')