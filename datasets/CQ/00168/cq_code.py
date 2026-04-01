import cadquery as cq

result = (
cq.Workplane("XY")
.polygon(5, 8)
.offset2D(1.5, kind='arc')
.extrude(3)
)
cq.exporters.export(result, 'GT.stl')