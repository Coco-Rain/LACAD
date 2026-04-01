import cadquery as cq

result = (
cq.Workplane()
.polarArray(25, 0, 360, 12)
.rect(1, 2)
.extrude(9)
.combine()
)
cq.exporters.export(result, 'GT.stl')